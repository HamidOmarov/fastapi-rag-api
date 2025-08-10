FROM python:3.11-slim
WORKDIR /app
ARG CACHEBUST=20250810
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    HUGGINGFACE_HUB_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache və data qovluqları
RUN mkdir -p /app/.cache /app/data/uploads /app/data/index && chmod -R 777 /app/.cache /app/data

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]

COPY . .
RUN mkdir -p /app/data/uploads /app/data/index
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
