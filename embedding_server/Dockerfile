# BGE-M3 embedding server (embedding_server/main.py). Separate image from
# the main app -- see requirements.txt in this directory for why (torch +
# FlagEmbedding are large, ML-specific, and nothing else needs them).
# Model weights (~2GB) download on first /embed request, not at build time
# -- first request after a cold start will be slow.
FROM python:3.11-slim

WORKDIR /app

# CPU torch wheel first -- pip's default Linux torch is the CUDA build
# (several GB of nvidia-* wheels) and compose runs this on CPU. For a GPU
# host, drop this line and add a GPU reservation to docker-compose.yml.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY embedding_server/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY embedding_server/main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
