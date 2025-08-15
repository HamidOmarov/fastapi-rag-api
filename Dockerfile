FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Sistəm asılılıqları (faiss, pypdf və s. üçün build-essential faydalıdır)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python asılılıqları
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    (pip install --no-cache-dir -r requirements.txt || true) && \
    pip install --no-cache-dir fastapi uvicorn[standard]

# Mənbə
COPY . .

# Default ENV-lər (kodunda istifadə olunursa)
ENV APP_ROOT=/app \
    DATA_DIR=/app/data \
    UPLOAD_DIR=/app/uploads \
    INDEX_DIR=/app/index \
    HF_HOME=/app/.cache \
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    OUTPUT_LANG=en

# HF $PORT-u verir; fallback 7860
EXPOSE 7860
CMD ["python","-u","boot.py"]
