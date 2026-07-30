"""Immutable, exportable audit log for guardrail triggers — CLAUDE.md #8:
"All guardrail triggers logged as immutable, exportable audit events."
AuditLogger is the seam the rest of query/ depends on. Every caller site
takes it as an optional dependency (default: don't log) rather than a
required one, so adding a new event type never breaks a caller that hasn't
wired a logger yet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    # "jailbreak_blocked" | "pii_redacted" | "permission_violation" |
    # "citation_stripped" | "flagged_for_review"
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str | None = None
    matter_id: str | None = None
    details: dict = Field(default_factory=dict)


class AuditLogger(Protocol):
    def log(self, event: AuditEvent) -> None: ...


class InMemoryAuditLogger:
    """In-memory sink -- the default in tests, and useful whenever a caller
    just needs to inspect events from within the same process run."""

    def __init__(self):
        self.events: list[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        self.events.append(event)


class JSONLAuditLogger:
    """Append-only JSON-lines file. Append-only is what makes this
    "immutable" in practice: nothing in this class ever rewrites, seeks
    into, or truncates the file -- every call opens in append mode and adds
    exactly one line."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
