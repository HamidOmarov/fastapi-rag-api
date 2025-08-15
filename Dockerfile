FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence-transformers \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Kod /app-da, data /workspace-da saxlanılacaq
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /workspace/data /workspace/uploads /workspace/index \
             /workspace/.cache/huggingface /workspace/.cache/sentence-transformers && \
    chmod -R 777 /workspace

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ENV-ləri /workspace-a yönəlt
ENV APP_ROOT=/workspace \
    DATA_DIR=/workspace/data \
    UPLOAD_DIR=/workspace/uploads \
    INDEX_DIR=/workspace/index \
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    OUTPUT_LANG=en \
    APP_MODULE=app.api:app

EXPOSE 7860
CMD ["python","-u","boot.py"]
