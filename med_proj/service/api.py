import os
import uuid
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from typing import Dict, Optional, List

from med_proj.service.schemas import (
    PredictRequest, PredictResponse, RagHit, HealthResponse,
    AllPredictResponse, ChatRequest, ChatResponse,
)
from med_proj.features.schema import CATEGORICAL, NUMERIC
from med_proj.rag.retrieve import retrieve
from med_proj.chatbot.engine import ChatEngine

ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

MODEL_PATHS = {
    "ed72": os.path.join(ART_DIR, "model_ed72.joblib"),
    "ed7d": os.path.join(ART_DIR, "model_ed7d.joblib"),
    "ed30d": os.path.join(ART_DIR, "model_ed30d.joblib"),
    "edadmit": os.path.join(ART_DIR, "model_ed_admit.joblib"),
}
RAG_INDEX_PATH = os.path.join(ART_DIR, "kb_index.joblib")

TASK_LABELS = {
    "ed72": "72-hour ED revisit",
    "ed7d": "7-day ED revisit",
    "ed30d": "30-day ED revisit",
    "edadmit": "ED-to-inpatient admission",
}

app = FastAPI(title="ED Revisit Risk API", version="1.0")
models = {}
_chat_sessions: Dict[str, ChatEngine] = {}

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def load_models():
    for k, p in MODEL_PATHS.items():
        if os.path.exists(p):
            models[k] = joblib.load(p)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = list(models.keys())
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        rag_index_loaded=os.path.exists(RAG_INDEX_PATH),
    )


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """Transform raw clinical inputs into the feature vector the model expects."""
    arr_hour = -1
    if req.ARRTIME is not None and req.ARRTIME >= 0:
        raw = int(req.ARRTIME)
        arr_hour = raw // 100 if raw <= 2359 else -1

    los_hours = 0.0
    if req.LOV is not None and req.LOV > 0:
        los_hours = req.LOV / 60.0

    encounter_dow = -1
    if req.VDAYR is not None and 1 <= req.VDAYR <= 7:
        encounter_dow = int(req.VDAYR) - 1  # 1=Sun->0, 7=Sat->6

    row = {
        "encounter_type": "ed",
        "prior_ed_30d": req.prior_ed_30d,
        "prior_ed_180d": req.prior_ed_180d,
        "days_since_last_encounter": req.days_since_last_encounter,
        "encounter_hour": arr_hour,
        "encounter_dow": encounter_dow,
        "los_hours": los_hours,
    }
    return pd.DataFrame([row])


def _auto_rag_query(task: str, prob: float, req: PredictRequest) -> str:
    """Build a contextual RAG query based on prediction results and patient data."""
    risk = "high" if prob > 0.3 else "moderate" if prob > 0.15 else "low"
    parts = [f"{risk} risk {TASK_LABELS.get(task, task)}"]

    if req.TOTCHRON is not None and req.TOTCHRON > 2:
        parts.append("multiple chronic conditions")
    if req.AGE is not None and req.AGE >= 65:
        parts.append("elderly patient")
    if req.PAINSCALE is not None and req.PAINSCALE >= 7:
        parts.append("severe pain")
    if req.SUBSTAB is not None and req.SUBSTAB == 1:
        parts.append("substance abuse")

    return "recommendations for " + " ".join(parts)


def _predict(task: str, req: PredictRequest) -> PredictResponse:
    if task not in models:
        return PredictResponse(task=task, probability=0.0, rag_hits=[])

    model = models[task]
    df = _build_feature_row(req)

    if isinstance(model, dict) and "model" in model:
        clf = model["model"]
    else:
        clf = model

    prob = float(clf.predict_proba(df)[:, 1][0])

    hits: List[RagHit] = []
    if os.path.exists(RAG_INDEX_PATH):
        query = req.question or _auto_rag_query(task, prob, req)
        raw_hits = retrieve(RAG_INDEX_PATH, query, top_k=4)
        hits = [
            RagHit(score=h["score"], source=h["source"], excerpt=h["excerpt"])
            for h in raw_hits
            if h["score"] > 0.05
        ]

    return PredictResponse(task=task, probability=prob, rag_hits=hits)


@app.post("/predict/ed72", response_model=PredictResponse)
def predict_ed72(req: PredictRequest):
    return _predict("ed72", req)


@app.post("/predict/ed7d", response_model=PredictResponse)
def predict_ed7d(req: PredictRequest):
    return _predict("ed7d", req)


@app.post("/predict/ed30d", response_model=PredictResponse)
def predict_ed30d(req: PredictRequest):
    return _predict("ed30d", req)


@app.post("/predict/edadmit", response_model=PredictResponse)
def predict_edadmit(req: PredictRequest):
    return _predict("edadmit", req)


@app.post("/predict/all", response_model=AllPredictResponse)
def predict_all(req: PredictRequest):
    results = {task: _predict(task, req) for task in MODEL_PATHS}
    return AllPredictResponse(predictions=results)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _chat_sessions:
        _chat_sessions[sid] = ChatEngine()
    engine = _chat_sessions[sid]
    reply = engine.respond(req.message)
    return ChatResponse(session_id=sid, reply=reply)


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
