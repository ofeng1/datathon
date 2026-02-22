from pydantic import BaseModel
from typing import List, Optional


class PredictRequest(BaseModel):
    """Raw clinical inputs from an ED visit (NHAMCS-style columns)."""
    AGE: Optional[float] = None
    SEX: Optional[float] = None
    RACERETH: Optional[float] = None
    PAYTYPER: Optional[float] = None
    IMMEDR: Optional[float] = None
    ARRTIME: Optional[float] = None
    WAITTIME: Optional[float] = None
    LOV: Optional[float] = None
    VDAYR: Optional[float] = None
    TEMPF: Optional[float] = None
    PULSE: Optional[float] = None
    RESPR: Optional[float] = None
    BPSYS: Optional[float] = None
    BPDIAS: Optional[float] = None
    PAINSCALE: Optional[float] = None
    TOTCHRON: Optional[float] = None
    INJURY: Optional[float] = None
    SUBSTAB: Optional[float] = None

    prior_ed_30d: int = 0
    prior_ed_180d: int = 0
    days_since_last_encounter: float = 999.0

    question: Optional[str] = None


class RagHit(BaseModel):
    score: float
    source: str
    excerpt: str


class PredictResponse(BaseModel):
    task: str
    probability: float
    rag_hits: List[RagHit] = []


class AllPredictResponse(BaseModel):
    predictions: dict[str, PredictResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    rag_index_loaded: bool


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
