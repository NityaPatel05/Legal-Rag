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
app's first ingest timed out after every cold start. Compose now gates the
app on this healthcheck (service_healthy), so the app never sees an unloaded
model.

Input limits. The server has no authentication (compose binds it to
loopback), but it must still not let one request exhaust memory: BGE-M3's
attention cost grows with sequence length, and an unbounded list of
8192-token texts is a trivial OOM. So:

- at most MAX_TEXTS_PER_REQUEST (256) texts per request -> 413 beyond;
  BGEM3Embedder batches client-side at 32, well under it;
- at most MAX_CHARS_PER_TEXT (32,000) characters per text -> 422; ~8k
  tokens at ~4 chars/token, i.e. the model's own ceiling;
- `max_length` (tokens) defaults to 2048 and may be raised per request up to
  8192, BGE-M3's limit. 2048 because of what ingestion embeds
  (ingestion/pipeline.embedding_text): a ~30-word document summary + a 1-2
  sentence context note + heading + one passage of at most MAX_PASSAGE_CHARS
  (1000 by default, hard-bounded by the splitter) -- about 400 tokens today.
  1024 would cover that, but R3's ablation runs MAX_PASSAGE_CHARS up to 4000
  (~1,000 tokens of passage alone + prefix), and a silently truncated
  passage tail is exactly the failure L1 fixed. 2048 covers every passage
  size in play with headroom; FlagEmbedding pads per batch to the longest
  text, so short passages don't pay for the higher cap.

Sparse output (return_sparse) stays off: server-side sparse retrieval is
roadmap R5 and changes the response contract on both sides.

The model revision is pinned (EMBEDDING_MODEL_REVISION, a Hugging Face
commit sha): an upstream re-upload of BAAI/bge-m3 would otherwise change
every new vector while old ones stay in Qdrant -- a silent mixed index.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_model = None
_model_lock = threading.Lock()
_load_error: str | None = None

MODEL_NAME = "BAAI/bge-m3"
# BAAI/bge-m3 `main` as of 2026-09-30 (HF API sha; last modified 2024-07-03).
_DEFAULT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_REVISION = os.environ.get("EMBEDDING_MODEL_REVISION", _DEFAULT_REVISION)

MODEL_MAX_TOKENS = 8192  # BGE-M3's hard limit
MAX_TEXTS_PER_REQUEST = int(os.environ.get("EMBED_MAX_TEXTS_PER_REQUEST", "256"))
MAX_CHARS_PER_TEXT = int(os.environ.get("EMBED_MAX_CHARS_PER_TEXT", "32000"))
DEFAULT_MAX_LENGTH = min(int(os.environ.get("BGE_M3_MAX_LENGTH", "2048")), MODEL_MAX_TOKENS)
_BATCH_SIZE = int(os.environ.get("BGE_M3_BATCH_SIZE", "16"))


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


def _get_model():
    global _model
    with _model_lock:  # warm-up thread and an early /embed must not both load
        if _model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel  # heavy import, off the import path

            # Download the pinned revision into the HF cache first, then load
            # from that local snapshot: BGEM3FlagModel doesn't take a
            # `revision` argument itself.
            from huggingface_hub import snapshot_download

            # The repo also carries an ONNX export (~2GB) and README images
            # this PyTorch path never reads.
            local_path = snapshot_download(
                MODEL_NAME,
                revision=MODEL_REVISION,
                ignore_patterns=["onnx/*", "imgs/*", "flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
            )
            # fp16 only on GPU: half precision on CPU is slow or unsupported for
            # some ops, and the compose file runs this on CPU by default.
            _model = BGEM3FlagModel(local_path, use_fp16=torch.cuda.is_available())
    return _model


class EmbedRequest(BaseModel):
    input: list[str]
    # Tokens per text; longer texts are truncated by the model.
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=1, le=MODEL_MAX_TOKENS)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
def health():
    if _model is not None:
        return {"status": "ok", "model": MODEL_NAME, "revision": MODEL_REVISION}
    status = {"status": "error", "error": _load_error} if _load_error else {"status": "loading"}
    return JSONResponse(status_code=503, content=status)


def _check_limits(request: EmbedRequest) -> None:
    if len(request.input) > MAX_TEXTS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"At most {MAX_TEXTS_PER_REQUEST} texts per request (got {len(request.input)}); batch client-side.",
        )
    for i, text in enumerate(request.input):
        if len(text) > MAX_CHARS_PER_TEXT:
            raise HTTPException(
                status_code=422,
                detail=f"input[{i}] is {len(text)} characters; the limit is {MAX_CHARS_PER_TEXT}.",
            )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    _check_limits(request)
    if not request.input:
        return EmbedResponse(embeddings=[])
    model = _get_model()
    output = model.encode(
        request.input,
        batch_size=_BATCH_SIZE,
        max_length=request.max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return EmbedResponse(embeddings=[vec.tolist() for vec in output["dense_vecs"]])
