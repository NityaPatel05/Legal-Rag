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
# uses. Pinned (same version as embedding_server/requirements.txt and the
# locally tested install) so a rebuild can't silently move it. The cache
# mount keeps a requirements.txt edit from re-downloading everything.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Non-root runtime user. The app parses counterparty-authored documents
# (docling) and talks to third-party APIs; if any of that is ever exploited,
# it should not be root in the container. Fixed UID so volume ownership is
# predictable. /app/audit and /app/reviews are created here, owned by the
# user, so the named volumes compose mounts there start out writable (Docker
# copies the image directory's ownership into an empty named volume).
# Upgrading a volume created by the old root image: run once
#   docker compose run --rm --user root app chown -R app:app /app/audit /app/reviews
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app/audit /app/reviews \
    && chown -R app:app /app/audit /app/reviews

COPY ingestion/ ./ingestion/
COPY query/ ./query/
COPY app/ ./app/

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
