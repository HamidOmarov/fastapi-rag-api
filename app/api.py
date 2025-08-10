# app/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import traceback

from .rag_system import SimpleRAG, UPLOAD_DIR
from .schemas import AskRequest, AskResponse, UploadResponse, HistoryResponse, HistoryItem
from .store import add_history, get_history
from .utils import ensure_session, http400

app = FastAPI(title="RAG API", version="1.0.0")

# CORS (Streamlit/UI üçün)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod-da domeninlə dəyiş
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            http400("Only PDF files are accepted.")
        dest = UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks_added = rag.add_pdf(dest)
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server xətası: {str(e)}"})

@app.post("/ask_question", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    try:
        session_id = ensure_session(payload.session_id)
        add_history(session_id, "user", payload.question)
        results = rag.search(payload.question, k=payload.top_k)
        contexts = [c for c, _ in results]
        answer = rag.synthesize_answer(payload.question, contexts) if hasattr(rag, "synthesize_answer") else None
        if answer is None:
            # rag_system.synthesize_answer funksiyasını birbaşa import etməmişiksə:
            from .rag_system import synthesize_answer
            answer = synthesize_answer(payload.question, contexts)
        add_history(session_id, "assistant", answer)
        return AskResponse(answer=answer, contexts=contexts, session_id=session_id)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server xətası: {str(e)}"})

@app.get("/get_history", response_model=HistoryResponse)
async def get_history_endpoint(session_id: str):
    try:
        hist_raw = get_history(session_id)
        history = [HistoryItem(**h) for h in hist_raw]
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server xətası: {str(e)}"})
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")
