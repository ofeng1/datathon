import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from med_proj.service.schemas import (
    HealthResponse, ChatRequest, ChatResponse,
)
from med_proj.chatbot.engine import ChatEngine

ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
RAG_INDEX_PATH = os.path.join(ART_DIR, "kb_index.joblib")
STATS_PATH = os.path.join(ART_DIR, "stats.json")

app = FastAPI(title="ED Revisit Risk API", version="1.0")
_chat_sessions: Dict[str, ChatEngine] = {}
_model_ok = False

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    global _model_ok
    try:
        engine = ChatEngine()
        _model_ok = bool(engine.models)
    except Exception:
        _model_ok = False


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = ["readmission"] if _model_ok else []
    return HealthResponse(
        status="ok" if _model_ok else "degraded",
        models_loaded=loaded,
        rag_index_loaded=os.path.exists(RAG_INDEX_PATH),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _chat_sessions:
        _chat_sessions[sid] = ChatEngine()
    engine = _chat_sessions[sid]
    reply = engine.respond(req.message)
    return ChatResponse(session_id=sid, reply=reply)


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """Return precomputed stats by region and condition for the Stats tab."""
    if not os.path.exists(STATS_PATH):
        return {"regions": [], "conditions": [], "national": {}}
    with open(STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
