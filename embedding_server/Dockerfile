# BGE-M3 embedding server (embedding_server/main.py). Separate image from
# the main app -- see requirements.txt in this directory for why (torch +
# FlagEmbedding are large, ML-specific, and nothing else needs them).
# Model weights (~2GB) download at first startup (background warm-up), not at
# build time, into the hf_cache volume.
FROM python:3.11-slim

WORKDIR /app

# CPU torch wheel first -- pip's default Linux torch is the CUDA build
# (several GB of nvidia-* wheels) and compose runs this on CPU. For a GPU
# host, drop this line and add a GPU reservation to docker-compose.yml.
# Pinned to the same version as requirements.txt (`==2.8.0` matches the
# `2.8.0+cpu` local version, so pip keeps this wheel).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY embedding_server/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Non-root runtime user (fixed UID). The HF cache directory is created and
# owned by the user so the hf_cache named volume mounted there starts out
# writable. Upgrading a volume created by the old root image (mounted at
# /root/.cache/huggingface): it is only a cache -- `docker volume rm` it, or
#   docker compose run --rm --user root embedding-server chown -R app:app /home/app/.cache
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

ENV HF_HOME=/home/app/.cache/huggingface

COPY embedding_server/main.py .

USER app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
