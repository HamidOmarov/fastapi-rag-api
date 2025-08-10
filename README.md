# FastAPI RAG API

**Live demo**
- API (HF Space): https://huggingface.co/spaces/HamidOmarov/FastAPI-RAG-API
- Dashboard (HF Space): https://huggingface.co/spaces/HamidOmarov/RAG-Dashboard

## What it does
Ask questions about your PDFs using vector search (FAISS) + sentence embeddings.
Robust to numeric/table-heavy docs, with AZ>EN translation and fallbacks.

## Quick test
curl -F "file=@sample.pdf" https://<api>/upload_pdf
curl -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the document about?\",\"top_k\":5}" https://<api>/ask_question

## Ops
- GET /health  - GET /stats  - POST /reset_index

## Stack
FastAPI · FAISS · sentence-transformers · pdfminer.six · Hugging Face Spaces
