"""Persistence for review cases, behind a Protocol like every other external
dependency. SQLite by default: the queue has to survive a restart (a flag
that vanishes on redeploy is the dead end H1 exists to remove), and SQLite
needs no extra service in docker-compose. Each case is stored as one JSON
document with the columns the queue listing filters on.

Concurrency. A decision used to be read -> check OPEN in Python -> INSERT OR
REPLACE the whole body, so two reviewers deciding at once both passed the
check and the last write won -- one released answer silently replaced
another, with two `review_decision` audit events. And opening a case
(first_viewed_at) rewrote the whole body too, so a GET that raced a
decision could put a RESOLVED case back to OPEN. The store now has two
narrow, conditional writes instead of a whole-body save:

- `record_decision`: `UPDATE ... WHERE review_id = ? AND status = 'open'`;
  0 rows -> `ReviewAlreadyDecided`. The check-and-set is one statement, so
  exactly one decision wins.
- `mark_viewed`: sets first_viewed_at only `WHERE ... first_viewed_at IS
  NULL AND status = 'open'`, touching nothing else in the body.

`save` stays for callers that still use it, but only dispatches to those two
writes -- it can no longer revert a decided case. An open case's body is
written once, by `add`.

`list(status="all")` (or None) returns every status; the HTTP route
defaults to "open".

Connections are opened per call and always closed (`contextlib.closing`):
`with sqlite3.connect(...)` alone only commits/rolls back, it never closes,
so the old code leaked a file handle per request.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter

from query.review.schemas import ReviewAlreadyDecided, ReviewCase, ReviewStatus

StatusFilter = ReviewStatus | Literal["all"] | None

_DATETIME = TypeAdapter(datetime)


def _status_value(status: StatusFilter) -> str | None:
    if status is None or status == "all":
        return None
    return ReviewStatus(status).value


class ReviewStore(Protocol):
    def add(self, case: ReviewCase) -> None: ...

    def get(self, review_id: str) -> ReviewCase | None: ...

    def mark_viewed(self, review_id: str, viewed_at: datetime) -> bool:
        """Sets first_viewed_at if it is unset and the case is open. True if
        this call set it."""
        ...

    def record_decision(self, case: ReviewCase) -> None:
        """Persists a resolved case iff the stored case is still open.
        Raises ReviewAlreadyDecided otherwise, LookupError if missing."""
        ...

    def save(self, case: ReviewCase) -> None: ...

    def list(self, *, tenant_id: str, matter_id: str, status: StatusFilter = None) -> list[ReviewCase]: ...


def _save_via_conditional_writes(store: ReviewStore, case: ReviewCase) -> None:
    if case.status == ReviewStatus.RESOLVED:
        store.record_decision(case)
    elif case.first_viewed_at is not None:
        store.mark_viewed(case.review_id, case.first_viewed_at)


class InMemoryReviewStore:
    """Same semantics as SQLiteReviewStore; the lock makes each
    check-and-set atomic across request threads."""

    def __init__(self):
        self._cases: dict[str, ReviewCase] = {}
        self._lock = threading.Lock()

    def add(self, case: ReviewCase) -> None:
        with self._lock:
            if case.review_id in self._cases:
                raise ValueError(f"Review {case.review_id} already exists")
            self._cases[case.review_id] = case.model_copy(deep=True)

    def get(self, review_id: str) -> ReviewCase | None:
        with self._lock:
            case = self._cases.get(review_id)
            return case.model_copy(deep=True) if case else None

    def mark_viewed(self, review_id: str, viewed_at: datetime) -> bool:
        with self._lock:
            stored = self._cases.get(review_id)
            if stored is None or stored.status != ReviewStatus.OPEN or stored.first_viewed_at is not None:
                return False
            stored.first_viewed_at = viewed_at
            return True

    def record_decision(self, case: ReviewCase) -> None:
        with self._lock:
            stored = self._cases.get(case.review_id)
            if stored is None:
                raise LookupError(f"Review {case.review_id} not found")
            if stored.status != ReviewStatus.OPEN:
                raise ReviewAlreadyDecided(f"Review {case.review_id} is already {stored.status.value}")
            self._cases[case.review_id] = case.model_copy(deep=True)

    def save(self, case: ReviewCase) -> None:
        _save_via_conditional_writes(self, case)

    def list(self, *, tenant_id: str, matter_id: str, status: StatusFilter = None) -> list[ReviewCase]:
        wanted = _status_value(status)
        with self._lock:
            return sorted(
                (
                    c.model_copy(deep=True)
                    for c in self._cases.values()
                    if c.tenant_id == tenant_id
                    and c.matter_id == matter_id
                    and (wanted is None or c.status.value == wanted)
                ),
                key=lambda c: c.created_at,
            )


class SQLiteReviewStore:
    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 30.0):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_seconds = busy_timeout_seconds
        with self._transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS review_cases ("
                " review_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, matter_id TEXT NOT NULL,"
                " status TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS review_cases_scope ON review_cases (tenant_id, matter_id, status)"
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One connection per call, committed (or rolled back) and closed."""
        with closing(sqlite3.connect(self._path, timeout=self._busy_timeout_seconds)) as conn:
            with conn:
                yield conn

    def add(self, case: ReviewCase) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?)",
                (
                    case.review_id,
                    case.tenant_id,
                    case.matter_id,
                    case.status.value,
                    case.created_at.isoformat(),
                    case.model_dump_json(),
                ),
            )

    def get(self, review_id: str) -> ReviewCase | None:
        with self._transaction() as conn:
            row = conn.execute("SELECT body FROM review_cases WHERE review_id = ?", (review_id,)).fetchone()
        return ReviewCase.model_validate_json(row[0]) if row else None

    def mark_viewed(self, review_id: str, viewed_at: datetime) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE review_cases SET body = json_set(body, '$.first_viewed_at', ?)"
                " WHERE review_id = ? AND status = ?"
                " AND json_extract(body, '$.first_viewed_at') IS NULL",
                (_DATETIME.dump_python(viewed_at, mode="json"), review_id, ReviewStatus.OPEN.value),
            )
            return cursor.rowcount == 1

    def record_decision(self, case: ReviewCase) -> None:
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE review_cases SET status = ?, body = ? WHERE review_id = ? AND status = ?",
                (case.status.value, case.model_dump_json(), case.review_id, ReviewStatus.OPEN.value),
            )
            if cursor.rowcount == 1:
                return
            row = conn.execute("SELECT status FROM review_cases WHERE review_id = ?", (case.review_id,)).fetchone()
        if row is None:
            raise LookupError(f"Review {case.review_id} not found")
        raise ReviewAlreadyDecided(f"Review {case.review_id} is already {row[0]}")

    def save(self, case: ReviewCase) -> None:
        _save_via_conditional_writes(self, case)

    def list(self, *, tenant_id: str, matter_id: str, status: StatusFilter = None) -> list[ReviewCase]:
        sql = "SELECT body FROM review_cases WHERE tenant_id = ? AND matter_id = ?"
        params: list[str] = [tenant_id, matter_id]
        wanted = _status_value(status)
        if wanted is not None:
            sql += " AND status = ?"
            params.append(wanted)
        with self._transaction() as conn:
            rows = conn.execute(sql + " ORDER BY created_at", params).fetchall()
        return [ReviewCase.model_validate_json(r[0]) for r in rows]
