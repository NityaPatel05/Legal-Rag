"""HTTP client for the contract-analysis API.

Runs inside the Streamlit server process, never in the browser: the API key
(RAG_API_KEY) is read from the server's environment and sent from here, so
it never reaches a viewer's machine. The flip side is that anyone who can
open the Streamlit page acts with that key -- the UI binds to localhost by
default (ui/.streamlit/config.toml); put real sign-in in front of it before
exposing it (CLAUDE.md R8).

Timeouts follow the backend's LLM call budget (CLAUDE.md #16): a query is at
most 2 + MAX_RETRIES LLM calls, an ingest 1 + ceil(passages / 8), and each
call may retry with backoff on a free-tier 429."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 30.0
QUERY_TIMEOUT = 300.0
INGEST_TIMEOUT = 1800.0

# Mirrors query/review/schemas.ID_PATTERN; checked here only to give an
# immediate message -- the API enforces it.
ID_PATTERN = r"^[A-Za-z0-9._-]{1,128}$"


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    matter_id: str

    @property
    def params(self) -> dict[str, str]:
        return {"tenant_id": self.tenant_id, "matter_id": self.matter_id}


class ApiError(Exception):
    """A failed call, with a message written for the person using the UI."""

    def __init__(self, status: int | None, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    @property
    def message(self) -> str:
        prefix = {
            401: "Not authenticated -- set RAG_API_KEY for the UI server.",
            403: "Not allowed.",
            404: "Not found.",
            409: "Conflict.",
            413: "Too large.",
            415: "Unsupported file.",
            422: "The API rejected the request.",
            501: "Not available on this server.",
            503: "A backend service is unavailable; try again shortly.",
        }.get(self.status or 0, "")
        return f"{prefix} {self.detail}".strip() if prefix else self.detail


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(detail, list):  # FastAPI request-validation errors
        parts = []
        for error in detail:
            field = ".".join(str(p) for p in error.get("loc", [])[1:])
            parts.append(f"{field}: {error.get('msg')}" if field else str(error.get("msg")))
        return "; ".join(parts)
    return str(detail or f"HTTP {response.status_code}")


class ApiClient:
    def __init__(self, http: httpx.Client):
        self._http = http

    @classmethod
    def from_env(cls) -> ApiClient:
        key = os.environ.get("RAG_API_KEY") or None
        return cls(
            httpx.Client(
                base_url=os.environ.get("RAG_API_URL", DEFAULT_API_URL),
                headers={"X-API-Key": key} if key else {},
                timeout=DEFAULT_TIMEOUT,
            )
        )

    @property
    def base_url(self) -> str:
        return str(self._http.base_url)

    @property
    def has_api_key(self) -> bool:
        return "x-api-key" in self._http.headers

    def _call(self, method: str, path: str, *, timeout: float | None = None, **kwargs) -> Any:
        try:
            response = self._http.request(
                method, path, timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT, **kwargs
            )
        except httpx.TimeoutException as exc:
            raise ApiError(None, "The API took too long to respond.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(None, f"Can't reach the API at {self.base_url} ({exc.__class__.__name__}).") from exc
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))
        return response.json()

    # --- documents ---

    def health(self) -> dict:
        return self._call("GET", "/health", timeout=5.0)

    def list_documents(self, scope: Scope) -> list[dict]:
        return self._call("GET", "/documents", params=scope.params)

    def get_document(self, scope: Scope, document_id: str) -> dict:
        return self._call("GET", f"/documents/{document_id}", params=scope.params)

    def ingest_text(self, scope: Scope, document_id: str, text: str, privilege_level: str) -> dict:
        body = {"document_id": document_id, "source_text": text, "privilege_level": privilege_level, **scope.params}
        return self._call("POST", "/documents/text", json=body, timeout=INGEST_TIMEOUT)

    def ingest_file(
        self, scope: Scope, document_id: str, filename: str, data: bytes, privilege_level: str
    ) -> dict:
        form = {"document_id": document_id, "privilege_level": privilege_level, **scope.params}
        return self._call(
            "POST", "/documents/file", data=form, files={"file": (filename, data)}, timeout=INGEST_TIMEOUT
        )

    # --- questions ---

    def query(
        self,
        scope: Scope,
        document_id: str,
        question: str,
        *,
        jurisdiction: str | None = None,
        governing_law: str | None = None,
    ) -> dict:
        body = {
            "document_id": document_id,
            "query_text": question,
            "jurisdiction": jurisdiction or None,
            "governing_law": governing_law or None,
            **scope.params,
        }
        return self._call("POST", "/query", json=body, timeout=QUERY_TIMEOUT)

    # --- human review ---

    def list_reviews(self, scope: Scope, status: str = "open") -> list[dict]:
        return self._call("GET", "/reviews", params={**scope.params, "status": status})

    def get_review(self, scope: Scope, review_id: str) -> dict:
        """Opening the case as its reviewer: starts the time-to-decision
        clock. An asker polling for an outcome uses review_status."""
        return self._call("GET", f"/reviews/{review_id}", params=scope.params)

    def review_status(self, scope: Scope, review_id: str) -> dict:
        return self._call("GET", f"/reviews/{review_id}/status", params=scope.params)

    def check_claims(self, scope: Scope, review_id: str, decisions: list[dict]) -> list[dict]:
        return self._call("POST", f"/reviews/{review_id}/check", json={**scope.params, "decisions": decisions})

    def decide(self, scope: Scope, review_id: str, reviewer: str, decisions: list[dict], note: str | None) -> dict:
        body = {**scope.params, "reviewer": reviewer, "decisions": decisions, "note": note or None}
        return self._call("POST", f"/reviews/{review_id}/decision", json=body)
