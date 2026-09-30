"""LLMClient is the seam every prompted-LLM step in ingestion/ depends on, so
metadata tagging and context enrichment are unit-testable with a fake client
and no network access. Production wiring is GroqLLMClient — see CLAUDE.md
'Tech stack' / .env.example for the GROQ_API_KEY / GROQ_MODEL scope decision.

Retry policy (review finding B21). Only failures a retry can fix are
retried: 429, 408/409, 5xx, timeouts/connection errors, and a response that
isn't a JSON object (JSON mode reduces but doesn't eliminate malformed
output). A 400/401/403/404/422 fails immediately -- the old loop retried
those too, burning ~14 s of backoff and three more requests on a bad API key.
A 429's Retry-After header is the server's own statement of when the window
reopens, so it's honoured (capped) in preference to the exponential
schedule; otherwise full-jitter exponential backoff, so parallel ingests
don't retry in lockstep.

CachingLLMClient (fewer calls). Every call runs at temperature 0, so the same
(system prompt, user prompt) is answered the same way; re-ingesting an
unchanged contract, resuming an ingest that died at chunk N, or re-asking a
question over the same evidence used to repeat every call. The cache is
keyed by a SHA-256 of (namespace, model, system, user) -- the namespace is
the tenant+matter, so a cache hit never reveals that another tenant sent the
same text -- and only successful, parsed responses are stored."""

from __future__ import annotations

import contextlib
import email.utils
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Runs a single prompted-LLM call and returns the parsed JSON response."""
        ...


class LLMCallError(RuntimeError):
    """Raised after every retry of an LLM call has failed, or at once on a
    non-retryable error (bad request, bad credentials)."""


_RETRYABLE_STATUS = frozenset({408, 409, 429})
_MAX_RETRY_AFTER_SECONDS = 60.0


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(exc: Exception) -> bool:
    """429 / 408 / 409 / 5xx, transport errors and malformed JSON retry;
    any other HTTP status is a caller error that a retry can't fix."""
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS or status >= 500
    # No HTTP status: timeout, connection reset, or a bad-JSON ValueError.
    return True


def retry_after_seconds(exc: Exception) -> float | None:
    """The Retry-After header of a 429/503, in seconds (delta or HTTP date)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value) if value else None
        if parsed is None:
            return None
        seconds = parsed.timestamp() - time.time()
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_SECONDS))


def _call_with_retries(call, *, max_attempts: int, initial_backoff_seconds: float, what: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            parsed = call()
            if not isinstance(parsed, dict):
                raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
            return parsed
        except Exception as exc:  # noqa: BLE001 -- classified below
            last_error = exc
            if not is_retryable(exc):
                logger.warning("%s: non-retryable error (%s), not retrying", what, _status_code(exc))
                break
            if attempt < max_attempts - 1:
                delay = retry_after_seconds(exc)
                if delay is None:
                    # Full jitter (AWS architecture blog, "Exponential Backoff And Jitter").
                    delay = random.uniform(0, initial_backoff_seconds * (2**attempt))
                logger.info("%s: attempt %d failed (%s); retrying in %.1fs", what, attempt + 1, exc, delay)
                time.sleep(delay)
    raise LLMCallError(f"{what}: LLM call failed") from last_error


class GroqLLMClient:
    """Groq-hosted inference (free tier). The `groq` SDK is imported lazily so
    importing this module never requires it to be installed.

    `reasoning_effort` ("low" | "medium" | "high", gpt-oss models) is optional:
    gpt-oss's hidden reasoning tokens count against tokens-per-minute, so a
    lower setting stretches the free tier; unset = provider default."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        *,
        max_attempts: int = 4,
        initial_backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        reasoning_effort: str | None = None,
    ):
        self._api_key = api_key
        self.model = model
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._timeout_seconds = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from groq import Groq  # lazy import

            self._client = Groq(api_key=self._api_key, timeout=self._timeout_seconds, max_retries=0)
        return self._client

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        client = self._client_instance()
        extra = {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}

        def call() -> dict:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                **extra,
            )
            return json.loads(response.choices[0].message.content)

        return _call_with_retries(
            call,
            max_attempts=self._max_attempts,
            initial_backoff_seconds=self._initial_backoff_seconds,
            what="groq",
        )


class OpenAICompatibleLLMClient:
    """Any OpenAI-compatible /chat/completions endpoint -- intended for a
    SELF-HOSTED model (vLLM, Ollama, llama.cpp server) that privileged
    documents may be sent to (roadmap R9; see pipeline.select_llm_client).
    Uses httpx directly so no extra SDK is needed. Same retry policy and
    JSON-object contract as GroqLLMClient."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        max_attempts: int = 4,
        initial_backoff_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
    ):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._timeout_seconds = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        import httpx  # lazy import

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        def call() -> dict:
            response = httpx.post(self._url, json=body, headers=headers, timeout=self._timeout_seconds)
            response.raise_for_status()  # HTTPStatusError carries .response -> status classified
            return json.loads(response.json()["choices"][0]["message"]["content"])

        return _call_with_retries(
            call,
            max_attempts=self._max_attempts,
            initial_backoff_seconds=self._initial_backoff_seconds,
            what="self-hosted LLM",
        )


# --- Response cache -------------------------------------------------------


class LLMResponseCache(Protocol):
    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, value: dict) -> None: ...


class InMemoryLLMCache:
    """Bounded LRU; for tests and single-process dev."""

    def __init__(self, max_entries: int = 10_000):
        self._max_entries = max_entries
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


class SQLiteLLMCache:
    """Persistent cache: survives restarts, so a re-ingest after a crash (or
    of an unchanged contract) replays stored responses instead of calling
    the provider. Only the key hash and the parsed response are stored --
    never the prompt. Entries older than `max_age_seconds` are ignored, so a
    model or prompt-quality change eventually flows through even for
    identical input."""

    def __init__(self, path: str, *, max_age_seconds: float = 30 * 24 * 3600):
        self._path = path
        self._max_age_seconds = max_age_seconds
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
            )

    @contextlib.contextmanager
    def _connect(self):
        with contextlib.closing(sqlite3.connect(self._path, timeout=30)) as conn:
            with conn:  # commit / rollback
                yield conn

    def get(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value, created FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is None or time.time() - row[1] > self._max_age_seconds:
            return None
        try:
            value = json.loads(row[0])
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, created) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )


def cache_key(namespace: str, system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha256()
    for part in (namespace, system_prompt, user_prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class CachingLLMClient:
    """Wraps an LLMClient with a response cache. `namespace` must identify
    both the model (a different model is a different answer) and the
    tenant+matter scope (see module docstring). Place it INSIDE any
    RedactingLLMClient so the key is computed over what is actually sent."""

    def __init__(self, inner: LLMClient, cache: LLMResponseCache, namespace: str):
        self._inner = inner
        self._cache = cache
        self._namespace = namespace

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        key = cache_key(self._namespace, system_prompt, user_prompt)
        try:
            cached = self._cache.get(key)
        except Exception:  # noqa: BLE001 -- a broken cache must never break a request
            logger.warning("LLM cache read failed", exc_info=True)
            cached = None
        if cached is not None:
            return dict(cached)
        result = self._inner.generate_json(system_prompt, user_prompt)
        try:
            self._cache.put(key, result)
        except Exception:  # noqa: BLE001
            logger.warning("LLM cache write failed", exc_info=True)
        return result


class CountingLLMClient:
    """Counts calls that reach the wrapped client -- used by tests and the
    eval harness to hold the per-ingest / per-query call budget."""

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls = 0

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        return self._inner.generate_json(system_prompt, user_prompt)
