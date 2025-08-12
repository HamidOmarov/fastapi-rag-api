# app/api.py
from __future__ import annotations

from typing import List, Optional
from collections import deque
from datetime import datetime
from time import perf_counter
import re
import os

import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .rag_system import SimpleRAG, UPLOAD_DIR, INDEX_DIR

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = FastAPI(title="RAG API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Lightweight stats store (in-memory)
# ------------------------------------------------------------------------------
class StatsStore:
    def __init__(self):
        self.documents_indexed = 0
        self.questions_answered = 0
        self.latencies_ms = deque(maxlen=500)
        # Mon..Sun simple counter (index 0 = today for simplicity)
        self.last7_questions = deque([0] * 7, maxlen=7)
        self.history = deque(maxlen=50)  # recent questions

    def add_docs(self, n: int):
        if n > 0:
            self.documents_indexed += n

    def add_question(self, latency_ms: Optional[int] = None, q: Optional[str] = None):
        self.questions_answered += 1
        if latency_ms is not None:
            self.latencies_ms.append(int(latency_ms))
        if len(self.last7_questions) < 7:
            self.last7_questions.appendleft(1)
        else:
            # attribute to "today" bucket
            self.last7_questions[0] += 1
        if q:
            self.history.appendleft(
                {"question": q, "timestamp": datetime.utcnow().isoformat()}
            )

    @property
    def avg_ms(self) -> int:
        return int(sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0

stats = StatsStore()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
_GENERIC_PATTERNS = [
    r"\bbased on document context\b",
    r"\bappears to be\b",
    r"\bgeneral (?:summary|overview)\b",
]

_STOPWORDS = {
    "the","a","an","of","for","and","or","in","on","to","from","with","by","is","are",
    "was","were","be","been","being","at","as","that","this","these","those","it",
    "its","into","than","then","so","such","about","over","per","via","vs","within"
}

def is_generic_answer(text: str) -> bool:
    if not text:
        return True
    low = text.strip().lower()
    if len(low) < 15:
        return True
    for pat in _GENERIC_PATTERNS:
        if re.search(pat, low):
            return True
    return False

def tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if w and w not in _STOPWORDS and len(w) > 2]

def extractive_answer(question: str, contexts: List[str], max_chars: int = 500) -> str:
    """
    Simple keyword-based extractive fallback:
    pick sentences containing most question tokens.
    """
    if not contexts:
        return "I couldn't find relevant information in the indexed documents for this question."

    q_tokens = set(tokenize(question))
    if not q_tokens:
        # if question is e.g. numbers only
        q_tokens = set(tokenize(" ".join(contexts[:1])))

    # split into sentences
    sentences: List[str] = []
    for c in contexts:
        c = c or ""
        # rough sentence split
        for s in re.split(r"(?<=[\.!\?])\s+|\n+", c.strip()):
            s = s.strip()
            if s:
                sentences.append(s)

    if not sentences:
        # fallback to first context chunk
        return (contexts[0] or "")[:max_chars]

    # score sentences
    scored: List[tuple[int, str]] = []
    for s in sentences:
        toks = set(tokenize(s))
        score = len(q_tokens & toks)
        scored.append((score, s))

    # pick top sentences with score > 0, otherwise first few sentences
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    picked: List[str] = []

    for score, sent in scored:
        if score <= 0 and picked:
            break
        if len(" ".join(picked) + " " + sent) > max_chars:
            break
        picked.append(sent)

    if not picked:
        # no overlap, take first ~max_chars from contexts
        return (contexts[0] or "")[:max_chars]

    return " ".join(picked).strip()

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "summarizer": "extractive_en + translate + fallback"}

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

    # retrieval
    try:
        hits = rag.search(q, k=k)  # expected: List[Tuple[str, float]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    contexts = [c for c, _ in (hits or []) if c] or (rag.last_added[:k] if getattr(rag, "last_added", None) else [])

    if not contexts:
        stats.add_question(int((perf_counter() - t0) * 1000), q=q)
        return AskResponse(
            answer="I couldn't find relevant information in the indexed documents for this question.",
            contexts=[]
        )

    # synthesis (LLM or rule-based inside rag)
    try:
        synthesized = rag.synthesize_answer(q, contexts) or ""
    except Exception:
        synthesized = ""

    # guard against generic/unchanging answers
    if is_generic_answer(synthesized):
        synthesized = extractive_answer(q, contexts, max_chars=600)

    latency_ms = int((perf_counter() - t0) * 1000)
    stats.add_question(latency_ms, q=q)
    return AskResponse(answer=synthesized.strip(), contexts=contexts)

@app.get("/get_history", response_model=HistoryResponse)
def get_history():
    return HistoryResponse(
        total_chunks=len(rag.chunks),
        history=[HistoryItem(**h) for h in list(stats.history)]
    )

@app.get("/stats")
def stats_endpoint():
    # keep backward compat fields + add dashboard-friendly metrics
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
        # also reset stats counters to avoid stale analytics
        stats.documents_indexed = 0
        stats.questions_answered = 0
        stats.latencies_ms.clear()
        stats.last7_questions = deque([0] * 7, maxlen=7)
        stats.history.clear()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
