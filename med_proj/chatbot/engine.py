"""Core chatbot engine — state management, prediction, RAG, response formatting."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from med_proj.chatbot.intents import (
    INTENT_ASSESS, INTENT_ASK, INTENT_GREETING,
    INTENT_HELP, INTENT_RESET, INTENT_UPDATE,
    classify,
)
from med_proj.chatbot.extractors import extract_all
from med_proj.rag.retrieve import retrieve

ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

_MODEL_KEYS = {
    "ed72": ("model_ed72.joblib", "72-hour ED revisit"),
    "ed7d": ("model_ed7d.joblib", "7-day ED revisit"),
    "ed30d": ("model_ed30d.joblib", "30-day ED revisit"),
    "edadmit": ("model_ed_admit.joblib", "ED-to-inpatient admission"),
}

RAG_INDEX_PATH = os.path.join(ART_DIR, "kb_index.joblib")

_FIELD_LABELS = {
    "AGE": "Age",
    "SEX": "Sex",
    "TEMPF": "Temp (°F)",
    "PULSE": "Pulse",
    "RESPR": "Resp rate",
    "BPSYS": "BP systolic",
    "BPDIAS": "BP diastolic",
    "PAINSCALE": "Pain scale",
    "ARRTIME": "Arrival time",
    "LOV": "Length of visit (min)",
    "VDAYR": "Day of week",
    "TOTCHRON": "Chronic conditions",
    "INJURY": "Injury",
    "SUBSTAB": "Substance abuse",
    "IMMEDR": "Triage acuity",
    "prior_ed_30d": "Prior ED visits (30d)",
    "days_since_last_encounter": "Days since last visit",
}


class ChatEngine:
    """Stateful, per-session chatbot engine."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        for key, (filename, _) in _MODEL_KEYS.items():
            path = os.path.join(ART_DIR, filename)
            if os.path.exists(path):
                self.models[key] = joblib.load(path)

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
            "Hello! I'm the ED Risk Assessment assistant.\n\n"
            "Describe a patient and I'll predict their risk of:\n"
            "  • 72-hour ED revisit\n"
            "  • 7-day ED revisit\n"
            "  • 30-day ED revisit\n"
            "  • Hospital admission\n\n"
            "Example: \"72 year old male, temp 101.2, BP 135/85, pulse 110, "
            "pain 8/10, 4 chronic conditions, been here 4 hours\"\n\n"
            "You can also ask clinical questions like \"What are the risk factors "
            "for ED revisits?\" and I'll search our knowledge base.\n\n"
            "Type 'help' for more options."
        )

    def _help(self) -> str:
        return (
            "Here's what I can do:\n\n"
            "ASSESS A PATIENT\n"
            "  Describe a patient in plain language. I'll extract the clinical\n"
            "  values and run all risk models. Examples:\n"
            "    • \"55 year old female, BP 90/60, pulse 120, pain 9/10\"\n"
            "    • \"patient age 30, male, triage 3, arrived at 2:30pm\"\n\n"
            "UPDATE VALUES\n"
            "  After an assessment, refine values:\n"
            "    • \"actually the pain is 5\"\n"
            "    • \"change age to 60\"\n\n"
            "ASK A QUESTION\n"
            "  Ask about clinical topics and I'll search the knowledge base:\n"
            "    • \"What are discharge planning best practices?\"\n"
            "    • \"Tell me about COPD and ED revisits\"\n\n"
            "OTHER COMMANDS\n"
            "  • 'new patient' or 'reset' — clear current patient data\n"
            "  • 'help' — show this message"
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

        parts = ["Here's what I found:\n"]
        for i, h in enumerate(relevant, 1):
            excerpt = h["excerpt"].strip()
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."
            parts.append(f"--- {h['source']} (relevance: {h['score']:.0%}) ---")
            parts.append(excerpt)
            parts.append("")

        return "\n".join(parts)

    def _assess(self, message: str) -> str:
        extracted = extract_all(message)
        if not extracted and not self.state:
            return (
                "I couldn't extract any clinical values from that. "
                "Try something like: \"65 year old male, temp 101, BP 140/90, "
                "pulse 105, pain 7/10, 3 chronic conditions\""
            )

        self.state.update(extracted)

        if not self.models:
            return "No models loaded. Run the training pipeline first."

        scores = self._run_predictions()

        parts: List[str] = []
        parts.append(self._format_patient_summary())
        parts.append("")
        parts.append(self._format_risk_scores(scores))

        rag_text = self._rag_recommendations(scores)
        if rag_text:
            parts.append("")
            parts.append(rag_text)

        parts.append("")
        parts.append(
            "You can update values (e.g. \"change pain to 3\"), "
            "ask a clinical question, or type 'new patient' to start over."
        )

        return "\n".join(parts)

    # ---- Prediction ----

    def _build_feature_row(self) -> pd.DataFrame:
        s = self.state
        arr_hour = -1
        arrtime = s.get("ARRTIME")
        if arrtime is not None and arrtime >= 0:
            raw = int(arrtime)
            arr_hour = raw // 100 if raw <= 2359 else -1

        los_hours = 0.0
        lov = s.get("LOV")
        if lov is not None and lov > 0:
            los_hours = lov / 60.0

        encounter_dow = -1
        vdayr = s.get("VDAYR")
        if vdayr is not None and 1 <= vdayr <= 7:
            encounter_dow = int(vdayr) - 1

        row = {
            "encounter_type": "ed",
            "prior_ed_30d": s.get("prior_ed_30d", 0),
            "prior_ed_180d": s.get("prior_ed_180d", 0),
            "days_since_last_encounter": s.get("days_since_last_encounter", 999.0),
            "encounter_hour": arr_hour,
            "encounter_dow": encounter_dow,
            "los_hours": los_hours,
        }
        return pd.DataFrame([row])

    def _run_predictions(self) -> Dict[str, float]:
        df = self._build_feature_row()
        scores: Dict[str, float] = {}
        for key, model in self.models.items():
            clf = model["model"] if isinstance(model, dict) and "model" in model else model
            prob = float(clf.predict_proba(df)[:, 1][0])
            scores[key] = prob
        return scores

    # ---- Response formatting ----

    def _format_patient_summary(self) -> str:
        if not self.state:
            return "No patient data recorded yet."

        lines = ["CURRENT PATIENT"]
        for field, label in _FIELD_LABELS.items():
            val = self.state.get(field)
            if val is not None:
                if field == "SEX":
                    display = "Male" if val == 1.0 else "Female"
                elif field == "INJURY":
                    display = "Yes" if val == 1.0 else "No"
                elif field == "SUBSTAB":
                    display = "Yes" if val == 1.0 else "No"
                else:
                    display = f"{val:g}" if isinstance(val, float) else str(val)
                lines.append(f"  {label}: {display}")

        return "\n".join(lines)

    def _format_risk_scores(self, scores: Dict[str, float]) -> str:
        lines = ["RISK ASSESSMENT"]
        for key, (_, label) in _MODEL_KEYS.items():
            prob = scores.get(key)
            if prob is None:
                continue
            pct = prob * 100
            bar = self._risk_bar(prob)
            level = "HIGH" if prob > 0.3 else "MODERATE" if prob > 0.15 else "LOW"
            lines.append(f"  {label}: {pct:.1f}% ({level}) {bar}")
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
        label = _MODEL_KEYS[max_task][1]
        risk = "high" if prob > 0.3 else "moderate" if prob > 0.15 else "low"

        query_parts = [f"{risk} risk {label} recommendations"]

        s = self.state
        if s.get("TOTCHRON", 0) > 2:
            query_parts.append("multiple chronic conditions")
        if s.get("AGE", 0) >= 65:
            query_parts.append("elderly patient")
        if s.get("PAINSCALE", 0) >= 7:
            query_parts.append("severe pain")
        if s.get("SUBSTAB", 0) == 1:
            query_parts.append("substance abuse")

        query = " ".join(query_parts)
        hits = retrieve(RAG_INDEX_PATH, query, top_k=2)
        relevant = [h for h in hits if h["score"] > 0.05]

        if not relevant:
            return ""

        lines = ["RECOMMENDATIONS"]
        for h in relevant:
            excerpt = h["excerpt"].strip()
            if len(excerpt) > 400:
                excerpt = excerpt[:400] + "..."
            lines.append(f"  [{h['source']}]")
            for text_line in excerpt.split("\n")[:8]:
                if text_line.strip():
                    lines.append(f"  {text_line.strip()}")
            lines.append("")

        return "\n".join(lines)
