# app/api.py
from typing import List

import faiss, os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from .rag_system import SimpleRAG, UPLOAD_DIR, INDEX_DIR

app = FastAPI(title="RAG API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

# ---------- Schemas ----------
class UploadResponse(BaseModel):
    filename: str
    chunks_added: int

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    contexts: List[str]

class HistoryResponse(BaseModel):
    total_chunks: int

# ---------- Utility ----------
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

# ---------- Core ----------
@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    added = rag.add_pdf(dest)
    if added == 0:
        # Clear message for scanned/empty PDFs
        raise HTTPException(status_code=400, detail="No extractable text found (likely a scanned image PDF).")
    return UploadResponse(filename=file.filename, chunks_added=added)

@app.post("/ask_question", response_model=AskResponse)
def ask_question(payload: AskRequest):
    hits = rag.search(payload.question, k=max(1, payload.top_k))
    contexts = [c for c, _ in hits]
    answer = rag.synthesize_answer(payload.question, contexts)
    return AskResponse(answer=answer, contexts=contexts or rag.last_added[:5])

@app.get("/get_history", response_model=HistoryResponse)
def get_history():
    return HistoryResponse(total_chunks=len(rag.chunks))

@app.get("/stats")
def stats():
    return {
        "total_chunks": len(rag.chunks),
        "faiss_ntotal": int(getattr(rag.index, "ntotal", 0)),
        "model_dim": int(getattr(rag.index, "d", rag.embed_dim)),
        "last_added_chunks": len(rag.last_added),
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
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
