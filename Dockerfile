# Main API app (app/main.py). Does NOT bundle BGE-M3 (FlagEmbedding/torch)
# -- embedding happens over HTTP against the embedding-server service (see
# docker-compose.yml and embedding_server/Dockerfile), so this image stays
# small and doesn't need GPU/ML runtime deps at all.
FROM python:3.11-slim

WORKDIR /app

# docling (document parsing) needs these system libs for PDF/image handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# docling (and sentence-transformers, for the optional NLI check) pull in
# torch. Install the CPU wheel first: pip's default Linux torch is the CUDA
# build, which drags in several GB of nvidia-* wheels this CPU image never
# uses. The cache mount keeps a requirements.txt edit from re-downloading
# everything.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ingestion/ ./ingestion/
COPY query/ ./query/
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
