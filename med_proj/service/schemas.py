from pydantic import BaseModel
from typing import List, Optional


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
