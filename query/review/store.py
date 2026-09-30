"""Persistence for review cases, behind a Protocol like every other external
dependency. SQLite by default: the queue has to survive a restart (a flag
that vanishes on redeploy is the dead end H1 exists to remove), and SQLite
needs no extra service in docker-compose. Each case is stored as one JSON
document with the columns the queue listing filters on."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from query.review.schemas import ReviewCase, ReviewStatus


class ReviewStore(Protocol):
    def add(self, case: ReviewCase) -> None: ...

    def get(self, review_id: str) -> ReviewCase | None: ...

    def save(self, case: ReviewCase) -> None: ...

    def list(self, *, tenant_id: str, matter_id: str, status: ReviewStatus | None = None) -> list[ReviewCase]: ...


class InMemoryReviewStore:
    def __init__(self):
        self._cases: dict[str, ReviewCase] = {}

    def add(self, case: ReviewCase) -> None:
        self._cases[case.review_id] = case.model_copy(deep=True)

    def get(self, review_id: str) -> ReviewCase | None:
        case = self._cases.get(review_id)
        return case.model_copy(deep=True) if case else None

    def save(self, case: ReviewCase) -> None:
        self._cases[case.review_id] = case.model_copy(deep=True)

    def list(self, *, tenant_id: str, matter_id: str, status: ReviewStatus | None = None) -> list[ReviewCase]:
        return sorted(
            (
                c.model_copy(deep=True)
                for c in self._cases.values()
                if c.tenant_id == tenant_id and c.matter_id == matter_id and (status is None or c.status == status)
            ),
            key=lambda c: c.created_at,
        )


class SQLiteReviewStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS review_cases ("
                " review_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, matter_id TEXT NOT NULL,"
                " status TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS review_cases_scope ON review_cases (tenant_id, matter_id, status)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _write(self, case: ReviewCase, sql: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                sql,
                (
                    case.review_id,
                    case.tenant_id,
                    case.matter_id,
                    case.status.value,
                    case.created_at.isoformat(),
                    case.model_dump_json(),
                ),
            )

    def add(self, case: ReviewCase) -> None:
        self._write(case, "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?)")

    def save(self, case: ReviewCase) -> None:
        self._write(case, "INSERT OR REPLACE INTO review_cases VALUES (?, ?, ?, ?, ?, ?)")

    def get(self, review_id: str) -> ReviewCase | None:
        with self._connect() as conn:
            row = conn.execute("SELECT body FROM review_cases WHERE review_id = ?", (review_id,)).fetchone()
        return ReviewCase.model_validate_json(row[0]) if row else None

    def list(self, *, tenant_id: str, matter_id: str, status: ReviewStatus | None = None) -> list[ReviewCase]:
        sql = "SELECT body FROM review_cases WHERE tenant_id = ? AND matter_id = ?"
        params: list[str] = [tenant_id, matter_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at", params).fetchall()
        return [ReviewCase.model_validate_json(r[0]) for r in rows]
