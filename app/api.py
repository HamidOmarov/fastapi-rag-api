# app/api.py
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from .rag_system import SimpleRAG, UPLOAD_DIR

app = FastAPI(title="RAG API", version="1.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

# ---------- Models ----------
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

# ---------- Debug ----------
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
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "summarizer": "extractive_en+translate+fallback"}

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
    return UploadResponse(filename=file.filename, chunks_added=added)

# app/api.py içində ask_question endpoint
@app.post("/ask_question", response_model=AskResponse)
def ask_question(payload: AskRequest):
    hits = rag.search(payload.question, k=max(1, payload.top_k))
    contexts = [c for c, _ in hits]
    # fallback: (optional) burda da son faylı ötürmək olar; synthesize_answer onsuz da edir:
    answer = rag.synthesize_answer(payload.question, contexts)
    return AskResponse(answer=answer, contexts=contexts or rag.last_added[:5])

@app.get("/get_history", response_model=HistoryResponse)
def get_history():
    return HistoryResponse(total_chunks=len(rag.chunks))
