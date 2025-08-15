## 🚀 Available for Hire

**Specialties:** RAG Systems | PDF Processing | FastAPI | LangChain
**Rate:** $15–45/hour
**Contact:** [Upwork](https://www.upwork.com/) | [Email](mailto:you@example.com)

---
---
title: FastAPI RAG API
sdk: docker
pinned: false
---

# FastAPI RAG API

Minimal backend for RAG (FastAPI + FAISS + Sentence-Transformers).
Open /docs for the interactive API.

## Quick test
curl -F "file=@sample.pdf" https://<API>/upload_pdf
curl -H "Content-Type: application/json" -d "{\"question\":\"What is the document about?\",\"top_k\":5}" https://<API>/ask_question
