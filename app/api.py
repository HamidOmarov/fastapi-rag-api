# app/api.py
from __future__ import annotations

import os
import re
from collections import deque
from datetime import datetime, timezone
from time import perf_counter
from typing import List, Optional, Dict, Any

import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .rag_system import SimpleRAG, UPLOAD_DIR, INDEX_DIR

__version__ = "1.3.1"

app = FastAPI(title="RAG API", version=__version__)

# CORS (Streamlit UI üçün)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

# -------------------- Schemas --------------------
class UploadResponse(BaseModel):
    filename: str
    chunks_added: int

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    contexts: List[str]

class HistoryItem(BaseModel):
    question: str
    timestamp: str

class HistoryResponse(BaseModel):
    total_chunks: int
    history: List[HistoryItem] = []

# -------------------- Stats (in-memory) --------------------
class StatsStore:
    def __init__(self):
        self.documents_indexed = 0
        self.questions_answered = 0
        self.latencies_ms = deque(maxlen=500)
        self.last7_questions = deque([0] * 7, maxlen=7)  # sadə günlük sayğac
        self.history = deque(maxlen=50)

    def add_docs(self, n: int):
        if n > 0:
            self.documents_indexed += int(n)

    def add_question(self, latency_ms: Optional[int] = None, q: Optional[str] = None):
        self.questions_answered += 1
        if latency_ms is not None:
            self.latencies_ms.append(int(latency_ms))
        if len(self.last7_questions) == 7:
            self.last7_questions[0] += 1
        if q:
            self.history.appendleft(
                {"question": q, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            )

    @property
    def avg_ms(self) -> int:
        return int(sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0

stats = StatsStore()

# -------------------- Helpers --------------------
_STOPWORDS = {
    "the","a","an","of","for","and","or","in","on","to","from","with","by","is","are",
    "was","were","be","been","being","at","as","that","this","these","those","it","its",
    "into","than","then","so","such","about","over","per","via","vs","within"
}

def _tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if w and w not in _STOPWORDS and len(w) > 2]

def _is_generic_answer(text: str) -> bool:
    if not text:
        return True
    low = text.strip().lower()
    if len(low) < 15:
        return True
    # tipik generik pattern-lər
    if "based on document context" in low or "appears to be" in low:
        return True
    return False

def _extractive_fallback(question: str, contexts: List[str], max_chars: int = 600) -> str:
    """ Sualın açar sözlərinə əsasən kontekstdən cümlələr seç. """
    if not contexts:
        return "I couldn't find relevant information in the indexed documents for this question."
    qtok = set(_tokenize(question))
    if not qtok:
        return (contexts[0] or "")[:max_chars]

    # cümlələrə böl və skorla
    sentences: List[str] = []
    for c in contexts:
        for s in re.split(r"(?<=[\.!\?])\s+|\n+", (c or "").strip()):
            s = s.strip()
            if s:
                sentences.append(s)

    scored: List[tuple[int, str]] = []
    for s in sentences:
        st = set(_tokenize(s))
        scored.append((len(qtok & st), s))
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    picked: List[str] = []
    for sc, s in scored:
        if sc <= 0 and picked:
            break
        if len((" ".join(picked) + " " + s).strip()) > max_chars:
            break
        picked.append(s)

    if not picked:
        return (contexts[0] or "")[:max_chars]
    bullets = "\n".join(f"- {p}" for p in picked)
    return f"Answer (based on document context):\n{bullets}"

# -------------------- Routes --------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "summarizer": "extractive_en + translate + keyword_fallback",
        "faiss_ntotal": int(getattr(rag.index, "ntotal", 0)),
        "model_dim": int(getattr(rag.index, "d", rag.embed_dim)),
    }

@app.get("/debug/translate")
def debug_translate():
    try:
        from transformers import pipeline
        tr = pipeline("translation", model="Helsinki-NLP/opus-mt-az-en", cache_dir=str(rag.cache_dir), device=-1)
        out = tr("Sənəd təmiri və quraşdırılması ilə bağlı işlər görülüb.", max_length=80)[0]["translation_text"]
        return {"ok": True, "example_out": out}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    added = rag.add_pdf(dest)
    if added == 0:
        raise HTTPException(status_code=400, detail="No extractable text found (likely a scanned image PDF).")

    stats.add_docs(added)
    return UploadResponse(filename=file.filename, chunks_added=added)

@app.post("/ask_question", response_model=AskResponse)
def ask_question(payload: AskRequest):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'question'.")

    k = max(1, int(payload.top_k))
    t0 = perf_counter()

    # 1) Həmişə sual embedding-i ilə axtar
    try:
        hits = rag.search(q, k=k)  # List[Tuple[text, score]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    contexts = [c for c, _ in (hits or []) if c] or (getattr(rag, "last_added", [])[:k] if getattr(rag, "last_added", None) else [])

    if not contexts:
        latency_ms = int((perf_counter() - t0) * 1000)
        stats.add_question(latency_ms, q=q)
        return AskResponse(
            answer="I couldn't find relevant information in the indexed documents for this question.",
            contexts=[]
        )

    # 2) Cavabı sintez et (rag içində LLM/rule-based ola bilər)
    try:
        synthesized = (rag.synthesize_answer(q, contexts) or "").strip()
    except Exception:
        synthesized = ""

    # 3) Generic görünürsə, extractive fallback
    if _is_generic_answer(synthesized):
        synthesized = _extractive_fallback(q, contexts, max_chars=600)

    latency_ms = int((perf_counter() - t0) * 1000)
    stats.add_question(latency_ms, q=q)
    return AskResponse(answer=synthesized, contexts=contexts)

@app.get("/get_history", response_model=HistoryResponse)
def get_history():
    return HistoryResponse(
        total_chunks=len(rag.chunks),
        history=[HistoryItem(**h) for h in list(stats.history)]
    )

@app.get("/stats")
def stats_endpoint():
    return {
        "documents_indexed": stats.documents_indexed,
        "questions_answered": stats.questions_answered,
        "avg_ms": stats.avg_ms,
        "last7_questions": list(stats.last7_questions),
        "total_chunks": len(rag.chunks),
        "faiss_ntotal": int(getattr(rag.index, "ntotal", 0)),
        "model_dim": int(getattr(rag.index, "d", rag.embed_dim)),
        "last_added_chunks": len(getattr(rag, "last_added", [])),
        "version": app.version,
    }

@app.post("/reset_index")
def reset_index():
    try:
        rag.index = faiss.IndexFlatIP(rag.embed_dim)
        rag.chunks = []
        rag.last_added = []
        for p in [INDEX_DIR / "faiss.index", INDEX_DIR / "meta.npy"]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        stats.documents_indexed = 0
        stats.questions_answered = 0
        stats.latencies_ms.clear()
        stats.last7_questions = deque([0] * 7, maxlen=7)
        stats.history.clear()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
