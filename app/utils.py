# app/utils.py
import uuid
from fastapi import HTTPException

def ensure_session(session_id: str | None) -> str:
    return session_id or str(uuid.uuid4())

def http400(msg: str):
    raise HTTPException(status_code=400, detail=msg)
