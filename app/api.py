# app/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pathlib import Path
import shutil
import traceback

from .rag_system import SimpleRAG, UPLOAD_DIR, synthesize_answer as summarize
from .schemas import AskRequest, AskResponse, UploadResponse, HistoryResponse, HistoryItem
from .store import add_history, get_history
from .utils import ensure_session, http400

app = FastAPI(title="RAG API", version="1.1.0")

# CORS (allow Streamlit or any origin; tighten later if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://HamidOmarov-RAG-Dashboard.hf.space"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

@app.get("/")
def root():
    # convenience: open docs instead of 404
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0", "summarizer": "extractive_en"}

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            http400("Only PDF files are accepted.")
        dest: Path = UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks_added = rag.add_pdf(dest)
        return UploadResponse(filename=file.filename, chunks_added=chunks_added)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

@app.post("/ask_question", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    try:
        session_id = ensure_session(payload.session_id)
        add_history(session_id, "user", payload.question)

        results = rag.search(payload.question, k=payload.top_k)
        contexts = [c for c, _ in results]

        # Always use the English extractive summarizer
        answer = summarize(payload.question, contexts)

        add_history(session_id, "assistant", answer)
        return AskResponse(answer=answer, contexts=contexts, session_id=session_id)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

@app.get("/get_history", response_model=HistoryResponse)
async def get_history_endpoint(session_id: str):
    try:
        hist_raw = get_history(session_id)
        history = [HistoryItem(**h) for h in hist_raw]
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})
