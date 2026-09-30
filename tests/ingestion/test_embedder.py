"""BGEM3Embedder (B13/B14): client-side batching, retry only on transient
failures, and no HTTP call for an empty input."""

import io
import json
import socket
import urllib.error

import pytest

from ingestion.embedding import embedder as embedder_module
from ingestion.embedding.embedder import BGEM3Embedder, EmbeddingServiceError


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Stands in for urllib.request.urlopen. `failures` is a list of
    exceptions raised (in order) before requests start succeeding."""

    def __init__(self, failures=(), wrong_count=False):
        self.failures = list(failures)
        self.wrong_count = wrong_count
        self.batches: list[list[str]] = []
        self.timeouts: list[float] = []

    def __call__(self, request, timeout):
        self.timeouts.append(timeout)
        if self.failures:
            raise self.failures.pop(0)
        texts = json.loads(request.data.decode("utf-8"))["input"]
        self.batches.append(texts)
        vectors = [[float(len(t))] for t in texts]
        if self.wrong_count:
            vectors = vectors[:-1]
        return _Response(json.dumps({"embeddings": vectors}).encode("utf-8"))


def _http_error(code, headers=None):
    return urllib.error.HTTPError("http://x/embed", code, "err", headers or {}, None)


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(embedder_module.urllib.request, "urlopen", fake)
    return fake


def _embedder(**kwargs):
    sleeps: list[float] = []
    return BGEM3Embedder("http://embed:8080/", sleep=sleeps.append, **kwargs), sleeps


def test_empty_input_makes_no_http_call(server):
    embedder, _ = _embedder()
    assert embedder.embed([]) == []
    assert server.batches == [] and server.timeouts == []


def test_batches_and_preserves_order(server):
    embedder, _ = _embedder(batch_size=32)
    texts = ["x" * (i + 1) for i in range(70)]

    vectors = embedder.embed(texts)

    assert [len(b) for b in server.batches] == [32, 32, 6]
    assert vectors == [[float(i + 1)] for i in range(70)]


def test_default_batch_size_and_timeout(server):
    embedder, _ = _embedder()
    embedder.embed(["a"] * 33)
    assert [len(b) for b in server.batches] == [32, 1]
    assert set(server.timeouts) == {120.0}


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        socket.timeout("timed out"),
        ConnectionResetError("reset"),
        _http_error(503),
        _http_error(500),
        _http_error(429),
    ],
)
def test_transient_failures_are_retried(server, failure):
    server.failures = [failure, failure]
    embedder, sleeps = _embedder()

    assert embedder.embed(["a", "b"]) == [[1.0], [1.0]]
    assert len(sleeps) == 2
    assert all(0 <= s <= 30 for s in sleeps)


@pytest.mark.parametrize("code", [400, 404, 413, 422])
def test_client_errors_are_not_retried(server, code):
    server.failures = [_http_error(code)]
    embedder, sleeps = _embedder()

    with pytest.raises(urllib.error.HTTPError):
        embedder.embed(["a"])
    assert sleeps == []


def test_gives_up_after_max_retries(server):
    server.failures = [urllib.error.URLError("down")] * 10
    embedder, sleeps = _embedder(max_retries=3)

    with pytest.raises(urllib.error.URLError):
        embedder.embed(["a"])
    assert len(sleeps) == 3


def test_backoff_grows_and_is_capped(monkeypatch, server):
    monkeypatch.setattr(embedder_module.random, "uniform", lambda lo, hi: hi)  # worst case of the jitter
    server.failures = [urllib.error.URLError("down")] * 7
    embedder, sleeps = _embedder(max_retries=7)

    embedder.embed(["a"])

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_retry_after_header_is_honored(server):
    server.failures = [_http_error(429, {"Retry-After": "7"})]
    embedder, sleeps = _embedder()
    embedder.embed(["a"])
    assert sleeps == [7.0]


def test_only_the_failed_batch_is_retried(monkeypatch, server):
    calls = {"n": 0}

    def flaky(request, timeout):
        calls["n"] += 1
        if calls["n"] == 2:  # the second batch's first attempt fails
            raise urllib.error.URLError("blip")
        return server(request, timeout)

    monkeypatch.setattr(embedder_module.urllib.request, "urlopen", flaky)
    embedder, _ = _embedder(batch_size=2)

    vectors = embedder.embed(["a", "bb", "ccc", "dddd"])

    assert calls["n"] == 3
    assert server.batches == [["a", "bb"], ["ccc", "dddd"]]
    assert vectors == [[1.0], [2.0], [3.0], [4.0]]


def test_wrong_vector_count_raises(server):
    server.wrong_count = True
    embedder, _ = _embedder()
    with pytest.raises(EmbeddingServiceError):
        embedder.embed(["a", "b"])


def test_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        BGEM3Embedder("http://x", batch_size=0)
