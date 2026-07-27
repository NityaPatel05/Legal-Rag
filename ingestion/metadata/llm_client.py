"""LLMClient is the seam every prompted-LLM step in ingestion/ depends on, so
metadata tagging and context enrichment are unit-testable with a fake client
and no network access. Production wiring is GroqLLMClient — see CLAUDE.md
'Tech stack' / .env.example for the GROQ_API_KEY / GROQ_MODEL scope decision."""

from __future__ import annotations

import json
from typing import Protocol


class LLMClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Runs a single prompted-LLM call and returns the parsed JSON response."""
        ...


class GroqLLMClient:
    """Groq-hosted inference (free tier). The `groq` SDK is imported lazily so
    importing this module never requires it to be installed."""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from groq import Groq  # lazy import

            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        client = self._client_instance()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        return json.loads(content)
