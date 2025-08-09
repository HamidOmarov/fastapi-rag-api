---
title: FastAPI RAG API
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FastAPI RAG API

Production-ready sadə RAG API.

## Endpoints
- `GET /health` — status
- `POST /upload_pdf` — multipart PDF yüklə
- `POST /ask_question` — `{question, session_id?, top_k}` ilə soruş
- `GET /get_history?session_id=...` — söhbət tarixçəsi

## Lokal işə salma
```bash
uvicorn app.api:app --reload
