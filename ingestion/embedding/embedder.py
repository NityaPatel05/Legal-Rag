"""EmbeddingClient is the seam pipeline.py depends on, so it's unit-testable
without a running BGE-M3 endpoint. BGEM3Embedder is the production adapter —
BGE-M3 is self-hosted per CLAUDE.md, reachable over HTTP at BGE_M3_ENDPOINT."""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""
        ...


class BGEM3Embedder:
    def __init__(self, endpoint: str, timeout_seconds: float = 30.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["embeddings"]
