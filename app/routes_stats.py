from fastapi import APIRouter
from .metrics import tracker

router = APIRouter()

@router.get("/stats")
def get_stats():
    return tracker.get_stats()

@router.get("/get_history")
def get_history():
    s = tracker.get_stats()
    return {"history": s["lastN_questions"], "total_chunks": s["total_chunks"]}

