FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/.cache TRANSFORMERS_CACHE=/app/.cache HUGGINGFACE_HUB_CACHE=/app/.cache SENTENCE_TRANSFORMERS_HOME=/app/.cache PYTHONUNBUFFERED=1
RUN mkdir -p /app/.cache /app/data/uploads /app/data/index && chmod -R 777 /app/.cache /app/data
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","7860"]
