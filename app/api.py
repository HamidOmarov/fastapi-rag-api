from app.storage import DATA_DIR, INDEX_DIR, HISTORY_JSON

﻿from app.storage import DATA_DIR, INDEX_DIR, HISTORY_JSON



# app/api.py


import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .rag_system import SimpleRAG, UPLOAD_DIR, INDEX_DIR

__version__ = "1.3.2"

app = FastAPI(title="RAG API", version=__version__)

# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ CORS в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ Core singleton & metrics в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
rag = SimpleRAG()

METRICS: Dict[str, Any] = {
    "questions_answered": 0,
    "avg_ms": 0.0,
    "last7_questions": [5, 8, 12, 7, 15, 11, 9],  # placeholder sample
    "last_added_chunks": 0,
}
HISTORY: List[Dict[str, Any]] = []  # [{"question":..., "timestamp":...}]

# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ Models в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
    total_chunks: int

class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = 5
    # Optional routing hint: "all" (default) or "last"
    scope: str = Field(default="all", pattern="^(all|last)$")

class AskResponse(BaseModel):
    answer: str
    contexts: List[str]
    used_top_k: int

class HistoryResponse(BaseModel):
    total_chunks: int
    history: List[Dict[str, Any]]

# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ Routes в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "summarizer": "extractive_en + translate + keyword_fallback",
        "faiss_ntotal": getattr(rag.index, "ntotal", 0),
        "model_dim": getattr(rag, "embed_dim", None),
    }

@app.get("/debug/translate")
def debug_translate():
    """
    Simple smoke test for the AZв†’EN translator pipeline (if available).
    """
    try:
        from transformers import pipeline  # type: ignore
        tr = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-az-en",
            cache_dir=str(rag.cache_dir),
            device=-1,
        )
        out = tr("SЙ™nЙ™d tЙ™miri vЙ™ quraЕџdД±rД±lmasД± ilЙ™ baДџlД± iЕџlЙ™r gГ¶rГјlГјb.", max_length=80)[0]["translation_text"]
        return {"ok": True, "example_out": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/upload_pdf", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF, extracts text, embeds, and adds to FAISS index.
    """
    name = file.filename or "uploaded.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    dest = UPLOAD_DIR / name
    try:
        # Save whole file to disk
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file.")
        dest.write_bytes(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")

    try:
        added = rag.add_pdf(dest)
        if added == 0:
            raise HTTPException(status_code=400, detail="No extractable text found (likely a scanned PDF).")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    METRICS["last_added_chunks"] = int(added)
    return UploadResponse(
        message="indexed",
        filename=name,
        chunks_added=added,
        total_chunks=len(rag.chunks),
    )

@app.post("/ask_question", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Retrieves top_k contexts and synthesizes an extractive answer.
    Supports optional scope hint: "all" or "last".
    """
    q = (req.question or "").strip()
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Question is too short.")

    start = time.perf_counter()

    # Prefer calling with scope if rag_system supports it; otherwise fallback.
    try:
        pairs = rag.search(q, k=req.top_k, scope=req.scope)  # type: ignore[arg-type]
    except TypeError:
        pairs = rag.search(q, k=req.top_k)

    contexts = [t for (t, _) in pairs]
    answer = rag.synthesize_answer(q, contexts, max_sentences=4)

    # metrics
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    METRICS["questions_answered"] += 1
    n = METRICS["questions_answered"]
    METRICS["avg_ms"] = (METRICS["avg_ms"] * (n - 1) + elapsed_ms) / n

    # history (cap to last 200)
    HISTORY.append({
        "question": q,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if len(HISTORY) > 200:
        del HISTORY[: len(HISTORY) - 200]

    return AskResponse(answer=answer, contexts=contexts, used_top_k=int(req.top_k))

@app.get("/get_history", response_model=HistoryResponse)
def get_history():
    return {"total_chunks": len(rag.chunks), "history": HISTORY[-50:]}

@app.get("/stats")
def stats():
    return {
        "documents_indexed": len(list(UPLOAD_DIR.glob("*.pdf"))),
        "questions_answered": METRICS["questions_answered"],
        "avg_ms": round(float(METRICS["avg_ms"]), 2),
        "last7_questions": METRICS.get("last7_questions", []),
        "total_chunks": len(rag.chunks),
        "faiss_ntotal": getattr(rag.index, "ntotal", 0),
        "model_dim": getattr(rag, "embed_dim", None),
        "last_added_chunks": METRICS.get("last_added_chunks", 0),
        "version": __version__,
    }

@app.post("/reset_index")
def reset_index():
    try:
        rag.index = faiss.IndexFlatIP(rag.embed_dim)
        rag.chunks = []
        rag.last_added = []
        # remove persisted files if present
        (INDEX_DIR / "faiss.index").unlink(missing_ok=True)
        (INDEX_DIR / "meta.npy").unlink(missing_ok=True)
        # persist empty state
        rag._persist()
        return {"message": "index reset", "ntotal": getattr(rag.index, "ntotal", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def _ensure_dirs():
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        # HISTORY_JSON parent is DATA_DIR
        HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # boot-un dayanmasının qarşısını alaq
        pass
