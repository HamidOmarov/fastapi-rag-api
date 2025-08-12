---
title: FastAPI RAG API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: fastapi
app_file: app/api.py
pinned: false
---

# FastAPI RAG API

Minimal backend for RAG (FastAPI + FAISS + Sentence-Transformers).
Open `/docs` for the interactive API.

## Live
- API (HF Space): https://huggingface.co/spaces/HamidOmarov/FastAPI-RAG-API
- Dashboard (HF Space): https://huggingface.co/spaces/HamidOmarov/RAG-Dashboard

## What it does
Ask questions about your PDFs using vector search (FAISS) + sentence embeddings.
Robust to numeric/table-heavy docs, with optional AZ→EN translation and fallbacks.

## Quick test
curl -F "file=@sample.pdf" https://<API>/upload_pdf
curl -H "Content-Type: application/json" -d '{"question":"What is the document about?","top_k":5}' https://<API>/ask_question

## Ops
- GET /health • GET /stats • GET /get_history • POST /reset_index

## Stack
FastAPI · sentence-transformers · FAISS · pypdf · Hugging Face Spaces