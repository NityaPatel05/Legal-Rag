# BGE-M3 embedding server (embedding_server/main.py). Separate image from
# the main app -- see requirements.txt in this directory for why (torch +
# FlagEmbedding are large, ML-specific, and nothing else needs them).
# Model weights (~2GB) download on first /embed request, not at build time
# -- first request after a cold start will be slow.
FROM python:3.11-slim

WORKDIR /app

COPY embedding_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embedding_server/main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
