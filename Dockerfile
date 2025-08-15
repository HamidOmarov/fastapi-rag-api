FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Sən app modulunu dəqiqləşdirmək istəyirsənsə, HF Secrets-də APP_MODULE=app.main:app ver
ENV APP_ROOT=/app \
    DATA_DIR=/app/data \
    UPLOAD_DIR=/app/uploads \
    INDEX_DIR=/app/index \
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    OUTPUT_LANG=en

EXPOSE 7860
CMD ["python","-u","boot.py"]
