"""EmbeddingClient is the seam pipeline.py depends on, so it's unit-testable
without a running BGE-M3 endpoint. BGEM3Embedder is the production adapter —
BGE-M3 is self-hosted per CLAUDE.md, reachable over HTTP at BGE_M3_ENDPOINT.

Why batching and retries: ingestion embeds AFTER every per-clause LLM call
(tagging, enrichment, profile), and the original adapter sent the whole
document in ONE request with a 30s timeout. A contract that splits into a
few hundred 1000-char passages is minutes of CPU BGE-M3 work, so the single
request timed out and threw away all of the (rate-limited, paid) LLM work
before it. Now:

- texts go in batches of `batch_size` (default 32) per request, each well
  under the server's 256-texts-per-request cap (embedding_server/main.py);
- each request gets a 120s timeout (32 passages on CPU is seconds; the
  margin absorbs a cold cache or a busy host);
- a request is retried with exponential backoff + full jitter on transient
  failures only -- connection errors, timeouts, HTTP 429 and 5xx. Any other
  4xx (e.g. 413/422 from the server's input limits) is a bug in what we
  sent and fails immediately; retrying can't fix it.
- embed([]) returns [] without touching the network.
"""

from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol

DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 4
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""
        ...


class EmbeddingServiceError(RuntimeError):
    """The embedding server failed, or answered with the wrong shape."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):  # must precede URLError (subclass)
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError))


class BGEM3Embedder:
    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            if len(embeddings) != len(batch):
                raise EmbeddingServiceError(
                    f"embedding server returned {len(embeddings)} vectors for {len(batch)} texts"
                )
            vectors.extend(embeddings)
        return vectors

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return self._post(batch)
            except Exception as exc:  # noqa: BLE001 -- classified below
                if not _is_retryable(exc) or attempt >= self._max_retries:
                    raise
                self._sleep(self._backoff_seconds(attempt, exc))
                attempt += 1

    @staticmethod
    def _backoff_seconds(attempt: int, exc: BaseException) -> float:
        # Honor a numeric Retry-After on 429/503 when the server sends one.
        if isinstance(exc, urllib.error.HTTPError) and exc.headers is not None:
            retry_after = exc.headers.get("Retry-After")
            if retry_after and retry_after.strip().isdigit():
                return min(float(retry_after), _BACKOFF_CAP_SECONDS)
        # Full jitter (AWS Architecture Blog, "Exponential Backoff and Jitter").
        return random.uniform(0, min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2**attempt))

    def _post(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": batch}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["embeddings"]
