"""LLM client retry policy (review finding B21) and the response cache
(fewer LLM calls)."""

import pytest

from ingestion.metadata import llm_client as mod
from ingestion.metadata.llm_client import (
    CachingLLMClient,
    InMemoryLLMCache,
    LLMCallError,
    SQLiteLLMCache,
    _call_with_retries,
    is_retryable,
    retry_after_seconds,
)


class _Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.response = _Response(status, headers)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize("status,expected", [(400, False), (401, False), (403, False), (404, False), (422, False),
                                             (408, True), (409, True), (429, True), (500, True), (503, True)])
def test_only_transient_statuses_are_retryable(status, expected):
    assert is_retryable(_StatusError(status)) is expected


def test_transport_errors_and_bad_json_are_retryable():
    assert is_retryable(TimeoutError())
    assert is_retryable(ValueError("Expecting value"))


def test_bad_credentials_fail_after_one_attempt(no_sleep):
    calls = []

    def call():
        calls.append(1)
        raise _StatusError(401)

    with pytest.raises(LLMCallError):
        _call_with_retries(call, max_attempts=4, initial_backoff_seconds=2.0, what="t")
    assert len(calls) == 1 and no_sleep == []


def test_rate_limit_honours_retry_after(no_sleep):
    attempts = iter([_StatusError(429, {"retry-after": "7"}), None])

    def call():
        exc = next(attempts)
        if exc:
            raise exc
        return {"ok": True}

    assert _call_with_retries(call, max_attempts=4, initial_backoff_seconds=2.0, what="t") == {"ok": True}
    assert no_sleep == [7.0]


def test_retry_after_is_capped():
    assert retry_after_seconds(_StatusError(429, {"retry-after": "3600"})) == 60.0
    assert retry_after_seconds(_StatusError(429)) is None


def test_non_object_json_is_retried_then_fails(no_sleep):
    with pytest.raises(LLMCallError):
        _call_with_retries(lambda: ["a list"], max_attempts=3, initial_backoff_seconds=1.0, what="t")
    assert len(no_sleep) == 2
    assert all(0 <= d <= 2.0 for d in no_sleep)  # full jitter within the exponential cap


class _Counting:
    def __init__(self):
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt):
        self.calls += 1
        return {"n": self.calls}


def test_cache_hit_skips_the_call_and_is_namespaced():
    inner, cache = _Counting(), InMemoryLLMCache()
    a = CachingLLMClient(inner, cache, "tenant-a")
    assert a.generate_json("s", "u") == {"n": 1}
    assert a.generate_json("s", "u") == {"n": 1}
    assert inner.calls == 1
    # Same prompt, other tenant: no cross-tenant hit.
    assert CachingLLMClient(inner, cache, "tenant-b").generate_json("s", "u") == {"n": 2}


def test_failures_are_not_cached():
    class Flaky:
        calls = 0

        def generate_json(self, s, u):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise LLMCallError("down")
            return {"ok": True}

    client = CachingLLMClient(Flaky(), InMemoryLLMCache(), "ns")
    with pytest.raises(LLMCallError):
        client.generate_json("s", "u")
    assert client.generate_json("s", "u") == {"ok": True}


def test_sqlite_cache_persists_and_stores_no_prompt_text(tmp_path):
    path = str(tmp_path / "c.sqlite3")
    inner = _Counting()
    CachingLLMClient(inner, SQLiteLLMCache(path), "ns").generate_json("sys", "secret contract text")
    CachingLLMClient(inner, SQLiteLLMCache(path), "ns").generate_json("sys", "secret contract text")
    assert inner.calls == 1
    assert b"secret contract text" not in (tmp_path / "c.sqlite3").read_bytes()


def test_a_broken_cache_never_breaks_the_call():
    class Broken:
        def get(self, key):
            raise OSError("disk")

        def put(self, key, value):
            raise OSError("disk")

    inner = _Counting()
    assert CachingLLMClient(inner, Broken(), "ns").generate_json("s", "u") == {"n": 1}
