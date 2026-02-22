"""Core chatbot engine — state management, prediction, RAG, response formatting."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from med_proj.chatbot.intents import (
    INTENT_ASSESS, INTENT_ASK, INTENT_GREETING,
    INTENT_HELP, INTENT_RESET, INTENT_UPDATE,
    classify,
)
from med_proj.chatbot.extractors import extract_all
from med_proj.rag.retrieve import retrieve


ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
_MODEL_KEY = "readmission"
_MODEL_FILE = "readmission_model.json"

RAG_INDEX_PATH = os.path.join(ART_DIR, "kb_index.joblib")

_VITAL_LABELS = {
    "AGE": "Age",
    "SEX": "Sex",
    "TEMPF": "Temp (°F)",
    "PULSE": "Pulse",
    "RESPR": "Resp rate",
    "BPSYS": "BP systolic",
    "BPDIAS": "BP diastolic",
    "POPCT": "SpO₂ %",
    "PAINSCALE": "Pain scale",
    "ARRTIME": "Arrival time",
    "LOV": "Length of visit (min)",
    "VDAYR": "Day of week",
    "IMMEDR": "Triage acuity (ESI)",
    "TOTCHRON": "Chronic conditions",
    "prior_ed_30d": "Prior ED visits (30 d)",
    "days_since_last_encounter": "Days since last visit",
}

_CONDITION_LABELS = {
    "COPD": "COPD",
    "CHF": "Heart Failure (CHF)",
    "CAD": "Coronary Artery Disease",
    "ASTHMA": "Asthma",
    "CKD": "Chronic Kidney Disease",
    "ESRD": "End-Stage Renal Disease",
    "HTN": "Hypertension",
    "DIABTYP0": "Diabetes",
    "DIABTYP1": "Diabetes Type 1",
    "DIABTYP2": "Diabetes Type 2",
    "CANCER": "Cancer",
    "DEPRN": "Depression",
    "CEBVD": "Cerebrovascular Disease",
    "ALZHD": "Alzheimer's / Dementia",
    "HYPLIPID": "Hyperlipidemia",
    "OBESITY": "Obesity",
    "OSA": "Sleep Apnea",
    "OSTPRSIS": "Osteoporosis",
    "EDHIV": "HIV",
    "ETOHAB": "Alcohol Use Disorder",
    "SUBSTAB": "Substance Abuse",
    "INJURY": "Injury / Trauma",
}

_CONDITION_DEFAULT_TRIAGE = {
    "CHF":      2.0,
    "COPD":     3.0,
    "ESRD":     3.0,
    "CKD":      3.0,
    "CANCER":   3.0,
    "CEBVD":    2.0,
    "CAD":      2.0,
    "DIABTYP1": 3.0,
    "DIABTYP2": 3.0,
    "DIABTYP0": 3.0,
    "ASTHMA":   3.0,
    "EDHIV":    3.0,
    "ALZHD":    3.0,
}


def _clean_excerpt(raw: str, max_chars: int = 1200) -> str:
    """Trim a RAG excerpt at a line boundary and strip top-level headings."""
    text = raw.strip()
    lines = text.split("\n")
    cleaned: list[str] = []
    for ln in lines:
        # Skip the top-level `# Title` line at the very start
        if ln.startswith("# ") and not cleaned:
            continue
        cleaned.append(ln)
    text = "\n".join(cleaned).strip()

    if len(text) <= max_chars:
        return text

    # Cut at last complete line before the limit
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.5:
        cut = cut[:last_nl]

    # Strip trailing incomplete heading or orphaned bold markers
    result_lines = cut.rstrip().split("\n")
    while result_lines:
        tail = result_lines[-1].strip()
        if not tail or tail.startswith("#") and len(tail) < 6:
            result_lines.pop()
        elif tail.count("**") % 2 != 0:
            result_lines.pop()
        else:
            break

    return "\n".join(result_lines) + "\n..."


class ChatEngine:
    """Stateful, per-session chatbot engine."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}
        self._load_models()
        self._load_stats()

    def _load_models(self) -> None:
        path = os.path.join(ART_DIR, _MODEL_FILE)
        m = xgb.XGBClassifier(enable_categorical=True)
        m.load_model(path)
        self.models[_MODEL_KEY] = m

    def _load_stats(self) -> None:
        path = os.path.join(ART_DIR, "stats.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._stats = json.load(f)
            except Exception:
                self._stats = {}

    def respond(self, message: str) -> str:
        intent = classify(message)

        if intent == INTENT_GREETING:
            return self._greeting()
        if intent == INTENT_HELP:
            return self._help()
        if intent == INTENT_RESET:
            return self._reset()
        if intent == INTENT_ASK:
            return self._ask(message)
        if intent in (INTENT_ASSESS, INTENT_UPDATE):
            return self._assess(message)

        return self._ask(message)

    # ---- Intent handlers ----

    def _greeting(self) -> str:
        return (
            "### Welcome\n"
            "I'm the **ED Risk Assessment** assistant.\n\n"
            "Describe a patient and I'll predict their **readmission risk** "
            "and provide evidence-based recommendations.\n\n"
            "**Example:** *72 year old male with COPD and CHF, "
            "temp 101.2, BP 135/85, pulse 110, pain 8/10*\n\n"
            "You can also ask clinical questions — "
            "e.g. *\"What are the risk factors for ED revisits?\"*\n\n"
            "Type **help** for more options."
        )

    def _help(self) -> str:
        return (
            "### What I can do\n\n"
            "**Assess a patient** — describe them in plain language:\n"
            "- *55 year old female with CHF, BP 90/60, pulse 120, pain 9/10*\n"
            "- *patient age 30, male, COPD, triage 3*\n\n"
            "**Update values** — refine after an assessment:\n"
            "- *actually the pain is 5*\n"
            "- *change age to 60*\n\n"
            "**Ask a question** — search the knowledge base:\n"
            "- *What are discharge planning best practices?*\n"
            "- *Tell me about COPD and ED revisits*\n\n"
            "**Other commands**\n"
            "- *new patient* or *reset* — clear current patient data\n"
            "- *help* — show this message"
        )

    def _reset(self) -> str:
        self.state.clear()
        return "Patient data cleared. Describe a new patient to begin."

    def _ask(self, message: str) -> str:
        if not os.path.exists(RAG_INDEX_PATH):
            return "Knowledge base not available. Run the training pipeline first."

        hits = retrieve(RAG_INDEX_PATH, message, top_k=3)
        relevant = [h for h in hits if h["score"] > 0.05]

        if not relevant:
            return (
                "I couldn't find relevant information for that question. "
                "Try rephrasing, or ask about topics like ED revisits, "
                "discharge planning, chronic conditions, or triage acuity."
            )

        parts = ["### Knowledge Base Results\n"]
        for h in relevant:
            excerpt = _clean_excerpt(h["excerpt"], max_chars=1500)
            parts.append(excerpt)
            parts.append("")

        return "\n".join(parts)

    def _assess(self, message: str) -> str:
        extracted = extract_all(message)
        if not extracted and not self.state:
            return (
                "I couldn't extract any clinical values from that. "
                "Try something like: *65 year old male with COPD, temp 101, "
                "BP 140/90, pulse 105, pain 7/10*"
            )

        self.state.update(extracted)
        self._infer_missing()

        if not self.models:
            return "No models loaded. Run the training pipeline first."

        scores = self._run_predictions()

        parts: List[str] = []
        parts.append(self._format_patient_summary())
        parts.append("")
        parts.append(self._format_risk_scores(scores))

        condition_risk = self._format_condition_risk_section()
        if condition_risk:
            parts.append("")
            parts.append(condition_risk)

        rag_text = self._rag_recommendations(scores)
        if rag_text:
            parts.append("")
            parts.append(rag_text)

        parts.append("\n---")
        parts.append(
            "*Update values (e.g. \"change pain to 3\"), "
            "ask a clinical question, or type **new patient** to start over.*"
        )

        return "\n".join(parts)

    # ---- Clinical inference ----

    def _infer_missing(self) -> None:
        """Fill in clinically reasonable defaults from the data we have."""
        s = self.state

        if s.get("IMMEDR") is None:
            best_triage = 5.0
            for cond, default_esi in _CONDITION_DEFAULT_TRIAGE.items():
                if s.get(cond) == 1.0:
                    best_triage = min(best_triage, default_esi)

            age = s.get("AGE")
            if age is not None and age >= 75:
                best_triage = min(best_triage, 3.0)

            pulse = s.get("PULSE")
            if pulse is not None and pulse > 120:
                best_triage = min(best_triage, 2.0)

            bpsys = s.get("BPSYS")
            if bpsys is not None and bpsys < 90:
                best_triage = min(best_triage, 2.0)

            temp = s.get("TEMPF")
            if temp is not None and temp > 101.0:
                best_triage = min(best_triage, 3.0)

            popct = s.get("POPCT")
            if popct is not None and popct < 92:
                best_triage = min(best_triage, 2.0)

            if best_triage < 5.0:
                s["IMMEDR"] = best_triage
                s["_inferred_triage"] = True

        if s.get("SEEN72") is None:
            totchron = s.get("TOTCHRON", 0)
            if totchron and totchron >= 2:
                s["SEEN72"] = 1.0

    # ---- Prediction ----

    def _build_feature_row(self) -> pd.DataFrame:
        clf = self.models[_MODEL_KEY]
        feat_names = clf.get_booster().feature_names or []
        row = {f: np.nan for f in feat_names}

        for k, v in self.state.items():
            if k in row and v is not None:
                row[k] = v

        df = pd.DataFrame([row], columns=feat_names)
        df = df.replace([-9, -8, -7], np.nan)
        return df

    def _run_predictions(self) -> dict:
        clf = self.models[_MODEL_KEY]
        df = self._build_feature_row()

        booster = clf.get_booster()
        feat_names = booster.feature_names
        df = df.reindex(columns=feat_names)

        X_np = df.to_numpy(dtype=np.float32, copy=True)
        dm = xgb.DMatrix(X_np, feature_names=feat_names)
        base_prob = float(booster.predict(dm)[0])

        prob = self._clinical_risk_adjustment(base_prob)
        return {"readmission": prob}

    def _clinical_risk_adjustment(self, base_prob: float) -> float:
        """Adjust the model baseline using evidence-based risk factors.

        The XGBoost model relies heavily on categorical features (diagnosis
        codes, drug IDs) that aren't available from free-text input.  This
        layer uses published readmission-rate literature to shift the
        probability when specific conditions or abnormal vitals are present.
        """
        s = self.state
        log_odds = np.log(base_prob / (1.0 - base_prob + 1e-9))

        _CONDITION_RISK = {
            "CHF":      1.8,
            "COPD":     1.5,
            "CKD":      1.3,
            "ESRD":     1.6,
            "DIABTYP0": 0.9,
            "DIABTYP1": 1.1,
            "DIABTYP2": 1.0,
            "CANCER":   1.2,
            "CEBVD":    1.0,
            "CAD":      0.9,
            "DEPRN":    0.7,
            "ASTHMA":   0.7,
            "ALZHD":    0.8,
            "EDHIV":    0.7,
            "ETOHAB":   0.9,
            "SUBSTAB":  0.8,
            "HTN":      0.4,
            "HYPLIPID": 0.3,
            "OBESITY":  0.4,
        }
        for cond, shift in _CONDITION_RISK.items():
            if s.get(cond) == 1.0:
                log_odds += shift

        age = s.get("AGE")
        if age is not None:
            if age >= 80:
                log_odds += 1.2
            elif age >= 75:
                log_odds += 1.0
            elif age >= 65:
                log_odds += 0.7
            elif age < 5:
                log_odds += 0.4
            elif age < 18:
                log_odds += 0.2

        pulse = s.get("PULSE")
        if pulse is not None:
            if pulse > 130:
                log_odds += 0.7
            elif pulse > 120:
                log_odds += 0.5
            elif pulse > 100:
                log_odds += 0.3

        bpsys = s.get("BPSYS")
        if bpsys is not None:
            if bpsys < 80:
                log_odds += 0.8
            elif bpsys < 90:
                log_odds += 0.6
            elif bpsys > 200:
                log_odds += 0.6
            elif bpsys > 180:
                log_odds += 0.4

        temp = s.get("TEMPF")
        if temp is not None:
            if temp > 102.0:
                log_odds += 0.6
            elif temp > 101.0:
                log_odds += 0.4
            elif temp > 100.4:
                log_odds += 0.2

        respr = s.get("RESPR")
        if respr is not None:
            if respr > 30:
                log_odds += 0.6
            elif respr > 24:
                log_odds += 0.4

        popct = s.get("POPCT")
        if popct is not None:
            if popct < 88:
                log_odds += 0.8
            elif popct < 92:
                log_odds += 0.5

        pain = s.get("PAINSCALE")
        if pain is not None and pain >= 8:
            log_odds += 0.4

        triage = s.get("IMMEDR")
        if triage is not None:
            if triage <= 1:
                log_odds += 1.0
            elif triage <= 2:
                log_odds += 0.7
            elif triage == 3:
                log_odds += 0.4

        totchron = s.get("TOTCHRON", 0)
        if totchron:
            if totchron >= 4:
                log_odds += 1.2
            elif totchron >= 3:
                log_odds += 0.8
            elif totchron >= 2:
                log_odds += 0.4

        injury = s.get("INJURY")
        if injury == 1.0:
            log_odds += 0.3

        prob = 1.0 / (1.0 + np.exp(-log_odds))
        return float(np.clip(prob, 0.0, 1.0))

    # ---- Response formatting ----

    def _format_patient_summary(self) -> str:
        if not self.state:
            return "No patient data recorded yet."

        s = self.state
        lines = ["### Patient Summary\n"]

        vitals_lines = []
        for field, label in _VITAL_LABELS.items():
            val = s.get(field)
            if val is None:
                continue
            if field == "SEX":
                display = "Male" if val == 1.0 else "Female"
            elif field == "IMMEDR":
                esi_map = {1: "1 — Immediate", 2: "2 — Emergent", 3: "3 — Urgent",
                           4: "4 — Semi-urgent", 5: "5 — Non-urgent"}
                display = esi_map.get(int(val), str(int(val)))
                if s.get("_inferred_triage"):
                    display += " *(inferred)*"
            else:
                display = f"{val:g}" if isinstance(val, float) else str(val)
            vitals_lines.append(f"- **{label}:** {display}")

        if vitals_lines:
            lines.extend(vitals_lines)

        conditions = []
        for field, label in _CONDITION_LABELS.items():
            if s.get(field) == 1.0:
                conditions.append(label)

        if conditions:
            lines.append("")
            lines.append("**Conditions:** " + ", ".join(conditions))

        canonical_conds = s.get("CONDITIONS", [])
        if canonical_conds:
            lines.append("")
            lines.append("**Detected Conditions (NLP):** " + ", ".join(canonical_conds))

        return "\n".join(lines)

    def _format_risk_scores(self, scores: dict) -> str:
        prob = scores.get("readmission", 0.0)
        pct = prob * 100

        if pct >= 30:
            level, label = "high", "High"
        elif pct >= 15:
            level, label = "moderate", "Moderate"
        else:
            level, label = "low", "Low"

        bar = self._risk_bar(prob)

        return (
            f"### Readmission Risk: **{pct:.1f}%** — **{label}**\n\n"
            f"{bar}"
        )

    def _format_condition_risk_section(self) -> str:
        """Format risk categories tied to conditions the user has entered."""
        s = self.state
        condition_ids: List[str] = []
        for cond in _CONDITION_LABELS:
            if s.get(cond) == 1.0:
                condition_ids.append(cond)
        canonical = s.get("CONDITIONS")
        if isinstance(canonical, list):
            for c in canonical:
                if isinstance(c, str) and c not in condition_ids:
                    condition_ids.append(c)

        if not condition_ids:
            return ""

        cond_stats = {c["id"]: c for c in self._stats.get("conditions", [])}
        lines = ["### Risk categories for this patient\n"]
        lines.append(
            "Based on the **conditions you entered**, here are relevant stats from national ED data:\n"
        )
        for cid in condition_ids:
            label = _CONDITION_LABELS.get(cid, cid)
            stat = cond_stats.get(cid)
            if stat:
                lines.append(
                    f"- **{label}:** {stat['pct_72h_revisit']}% 72-hour ED revisit rate, "
                    f"{stat['pct_admitted']}% admitted (NHAMCS sample)."
                )
            else:
                lines.append(f"- **{label}:** (no aggregate stats in this dataset)")
        return "\n".join(lines)

    @staticmethod
    def _risk_bar(prob: float, width: int = 20) -> str:
        filled = int(round(prob * width))
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    def _rag_recommendations(self, scores: Dict[str, float]) -> str:
        if not os.path.exists(RAG_INDEX_PATH):
            return ""

        max_task = max(scores, key=scores.get) if scores else None
        if max_task is None:
            return ""

        prob = scores[max_task]
        risk = "high" if prob > 0.3 else "moderate" if prob > 0.15 else "low"

        query_parts = [f"{risk} risk readmission recommendations"]

        s = self.state
        for cond, label in _CONDITION_LABELS.items():
            if s.get(cond) == 1.0:
                query_parts.append(label)

        if s.get("AGE", 0) >= 65:
            query_parts.append("elderly patient")
        if s.get("PAINSCALE", 0) >= 7:
            query_parts.append("severe pain")

        query = " ".join(query_parts)
        hits = retrieve(RAG_INDEX_PATH, query, top_k=3)
        relevant = [h for h in hits if h["score"] > 0.03]

        if not relevant:
            return ""

        lines = ["### Recommendations\n"]
        for h in relevant:
            excerpt = _clean_excerpt(h["excerpt"], max_chars=1200)
            lines.append(excerpt)
            lines.append("")

        return "\n".join(lines)
