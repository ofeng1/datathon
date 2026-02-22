import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from med_proj.service.schemas import (
    HealthResponse, ChatRequest, ChatResponse, ParseEdDocumentResponse,
)
from med_proj.chatbot.engine import ChatEngine
from med_proj.data.ed_form_parser import parse_ed_form_text, pdf_to_text

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
    if req.merge_state:
        for k, v in req.merge_state.items():
            if v is not None:
                engine.state[k] = v
    reply = engine.respond(req.message)
    return ChatResponse(session_id=sid, reply=reply)


@app.post("/parse-ed-document", response_model=ParseEdDocumentResponse)
async def parse_ed_document(file: UploadFile = File(...)):
    """Accept a PDF ED record form; extract text and parse into structured state."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    try:
        text = pdf_to_text(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e!s}")
    parsed = parse_ed_form_text(text)
    # Build a short summary for the chat message
    parts = []
    if parsed.get("AGE") is not None:
        parts.append(f"{int(parsed['AGE'])}yo")
    if parsed.get("SEX") is not None:
        parts.append("male" if parsed["SEX"] == 1.0 else "female")
    if parsed.get("BPSYS") is not None and parsed.get("BPDIAS") is not None:
        parts.append(f"BP {int(parsed['BPSYS'])}/{int(parsed['BPDIAS'])}")
    if parsed.get("PULSE") is not None:
        parts.append(f"pulse {int(parsed['PULSE'])}")
    if parsed.get("TEMPF") is not None:
        parts.append(f"temp {parsed['TEMPF']}")
    conds = [k for k, v in parsed.items() if v == 1.0 and k in (
        "COPD", "CHF", "CAD", "ASTHMA", "CKD", "HTN", "DIABTYP0", "DIABTYP1", "DIABTYP2",
        "CANCER", "DEPRN", "CEBVD", "SUBSTAB", "INJURY",
    )]
    if conds:
        parts.append("conditions: " + ", ".join(conds))
    summary = "Patient from uploaded ED form: " + ", ".join(parts) if parts else "Patient from uploaded ED form (parsed fields loaded)."
    return ParseEdDocumentResponse(parsed=parsed, summary=summary)


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
