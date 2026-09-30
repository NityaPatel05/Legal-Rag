"""LLMClient is the seam every prompted-LLM step in ingestion/ depends on, so
metadata tagging and context enrichment are unit-testable with a fake client
and no network access. Production wiring is GroqLLMClient — see CLAUDE.md
'Tech stack' / .env.example for the GROQ_API_KEY / GROQ_MODEL scope decision."""

from __future__ import annotations

import json
import time
from typing import Protocol


class LLMClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Runs a single prompted-LLM call and returns the parsed JSON response."""
        ...


class LLMCallError(RuntimeError):
    """Raised after every retry of an LLM call has failed."""


class GroqLLMClient:
    """Groq-hosted inference (free tier). The `groq` SDK is imported lazily so
    importing this module never requires it to be installed.

    Retries with exponential backoff: ingestion makes ~2 calls per clause
    (tag + enrich), which is well past free-tier per-minute limits for any
    real contract, so a 429 is the expected case, not an edge case. A
    response that isn't a JSON object is retried too -- JSON mode reduces but
    does not eliminate malformed output."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        *,
        max_attempts: int = 4,
        initial_backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
    ):
        self._api_key = api_key
        self._model = model
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._timeout_seconds = timeout_seconds
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from groq import Groq  # lazy import

            self._client = Groq(api_key=self._api_key, timeout=self._timeout_seconds, max_retries=0)
        return self._client

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        client = self._client_instance()
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                parsed = json.loads(response.choices[0].message.content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
                return parsed
            except Exception as exc:  # noqa: BLE001 -- rate limits, timeouts, bad JSON all retry the same way
                last_error = exc
                if attempt < self._max_attempts - 1:
                    time.sleep(self._initial_backoff_seconds * (2**attempt))
        raise LLMCallError(f"LLM call failed after {self._max_attempts} attempts") from last_error


class OpenAICompatibleLLMClient:
    """Any OpenAI-compatible /chat/completions endpoint -- intended for a
    SELF-HOSTED model (vLLM, Ollama, llama.cpp server) that privileged
    documents may be sent to (roadmap R9; see pipeline.select_llm_client).
    Uses httpx directly so no extra SDK is needed. Same retry/backoff and
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
        self._model = model
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._timeout_seconds = timeout_seconds

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        import httpx  # lazy import

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = httpx.post(self._url, json=body, headers=headers, timeout=self._timeout_seconds)
                response.raise_for_status()
                parsed = json.loads(response.json()["choices"][0]["message"]["content"])
                if not isinstance(parsed, dict):
                    raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
                return parsed
            except Exception as exc:  # noqa: BLE001 -- same retry policy as GroqLLMClient
                last_error = exc
                if attempt < self._max_attempts - 1:
                    time.sleep(self._initial_backoff_seconds * (2**attempt))
        raise LLMCallError(f"LLM call failed after {self._max_attempts} attempts") from last_error
