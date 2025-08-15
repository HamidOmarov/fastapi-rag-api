# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    session_id: Optional[str] = None
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    contexts: List[str]
    session_id: str

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int

class HistoryItem(BaseModel):
    role: str
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    history: List[HistoryItem]
