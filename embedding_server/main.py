"""Minimal embedding server for BGE-M3, matching the exact HTTP contract
ingestion/embedding/embedder.py's BGEM3Embedder already expects: POST /embed
with {"input": [...]} -> {"embeddings": [[...], ...]}.

Written in-house rather than pointing BGEM3Embedder at a third-party
embedding-server image, so both sides of the contract are owned by this
repo and can't silently drift apart. BGE-M3 is a locked-in tech-stack
decision (CLAUDE.md); FlagEmbedding is already a project dependency for it.

The model loads in a background thread at startup, and /health returns 503
until it is ready. Loading lazily on the first /embed call (the original
design) meant that call spent minutes downloading ~2GB of weights and the
app's BGEM3Embedder (30s timeout) failed the first ingest after every cold
start. Compose now gates the app on this healthcheck (service_healthy), so
the app never sees an unloaded model.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_model = None
_model_lock = threading.Lock()
_load_error: str | None = None


def _warm_up() -> None:
    global _load_error
    try:
        _get_model()
    except Exception as exc:  # noqa: BLE001 -- surfaced via /health, not raised
        _load_error = repr(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="BGE-M3 Embedding Server", lifespan=lifespan)


# BGE-M3 accepts up to 8192 tokens; anything longer is silently truncated.
# Ingestion splits clauses into passages well under this (see
# clause_parser.split_oversize_clauses), so truncation should never happen.
_MAX_LENGTH = int(os.environ.get("BGE_M3_MAX_LENGTH", "8192"))
_BATCH_SIZE = int(os.environ.get("BGE_M3_BATCH_SIZE", "16"))


def _get_model():
    global _model
    with _model_lock:  # warm-up thread and an early /embed must not both load
        if _model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel  # heavy import, off the import path

            # fp16 only on GPU: half precision on CPU is slow or unsupported for
            # some ops, and the compose file runs this on CPU by default.
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())
    return _model


class EmbedRequest(BaseModel):
    input: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health():
    if _model is not None:
        return {"status": "ok"}
    status = {"status": "error", "error": _load_error} if _load_error else {"status": "loading"}
    return JSONResponse(status_code=503, content=status)


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    model = _get_model()
    output = model.encode(
        request.input,
        batch_size=_BATCH_SIZE,
        max_length=_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return EmbedResponse(embeddings=[vec.tolist() for vec in output["dense_vecs"]])
