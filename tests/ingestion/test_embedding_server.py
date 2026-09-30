"""embedding_server/main.py input limits (B23). The model is replaced by a
fake, so FlagEmbedding/torch aren't needed."""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MAIN = Path(__file__).resolve().parents[2] / "embedding_server" / "main.py"


class _Vec(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self):
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return {"dense_vecs": [_Vec([0.5, 0.5]) for _ in texts]}


@pytest.fixture
def server(monkeypatch):
    spec = importlib.util.spec_from_file_location("embedding_server_main_under_test", _MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = FakeModel()
    monkeypatch.setattr(module, "_get_model", lambda: fake)
    module._model = fake  # /health reports ready
    # No `with`: the lifespan warm-up thread would try to load the real model.
    return module, TestClient(module.app), fake


def test_embeds_with_default_max_length(server):
    module, client, fake = server
    response = client.post("/embed", json={"input": ["a", "b"]})
    assert response.status_code == 200
    assert response.json() == {"embeddings": [[0.5, 0.5], [0.5, 0.5]]}
    assert fake.calls[0]["max_length"] == module.DEFAULT_MAX_LENGTH == 2048
    assert fake.calls[0]["return_sparse"] is False  # R5 not started; contract unchanged


def test_too_many_texts_is_413(server):
    module, client, fake = server
    response = client.post("/embed", json={"input": ["a"] * (module.MAX_TEXTS_PER_REQUEST + 1)})
    assert response.status_code == 413
    assert fake.calls == []


def test_text_over_char_cap_is_422(server):
    module, client, fake = server
    response = client.post("/embed", json={"input": ["ok", "x" * (module.MAX_CHARS_PER_TEXT + 1)]})
    assert response.status_code == 422
    assert "input[1]" in response.json()["detail"]
    assert fake.calls == []


def test_max_length_is_capped_at_the_model_limit(server):
    module, client, fake = server
    assert client.post("/embed", json={"input": ["a"], "max_length": 8193}).status_code == 422
    assert client.post("/embed", json={"input": ["a"], "max_length": 0}).status_code == 422
    assert client.post("/embed", json={"input": ["a"], "max_length": 8192}).status_code == 200
    assert fake.calls[-1]["max_length"] == 8192


def test_empty_input_does_not_touch_the_model(server):
    _, client, fake = server
    assert client.post("/embed", json={"input": []}).json() == {"embeddings": []}
    assert fake.calls == []


def test_revision_is_pinned_to_a_commit_by_default(server):
    module, client, _ = server
    assert len(module._DEFAULT_REVISION) == 40 and all(c in "0123456789abcdef" for c in module._DEFAULT_REVISION)
    assert client.get("/health").json()["revision"] == module.MODEL_REVISION
