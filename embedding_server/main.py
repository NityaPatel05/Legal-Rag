"""Minimal embedding server for BGE-M3, matching the exact HTTP contract
ingestion/embedding/embedder.py's BGEM3Embedder already expects: POST /embed
with {"input": [...]} -> {"embeddings": [[...], ...]}.

Written in-house rather than pointing BGEM3Embedder at a third-party
embedding-server image, so both sides of the contract are owned by this
repo and can't silently drift apart. BGE-M3 is a locked-in tech-stack
decision (CLAUDE.md); FlagEmbedding is already a project dependency for it.

The model loads lazily on first request (not at import time) so `docker
compose up` doesn't block container startup on a ~2GB model download --
the first /embed call after a cold start will be slow, subsequent ones
won't.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BGE-M3 Embedding Server")

_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel  # heavy import, deferred to first request

        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


class EmbedRequest(BaseModel):
    input: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    model = _get_model()
    output = model.encode(
        request.input, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    return EmbedResponse(embeddings=[vec.tolist() for vec in output["dense_vecs"]])
