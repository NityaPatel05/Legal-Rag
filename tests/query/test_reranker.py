from unittest.mock import MagicMock

from query.retrieval.reranker import CohereReranker
from query.schemas import RetrievedClause
from tests.query.factories import make_clause, make_metadata


class _FakeRerankResult:
    def __init__(self, index, relevance_score):
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankResponse:
    def __init__(self, results):
        self.results = results


def _reranker_with_fake_client(fake_client) -> CohereReranker:
    reranker = CohereReranker(api_key="fake-key")
    reranker._client = fake_client  # bypass the lazy `cohere` import entirely
    return reranker


def test_rerank_reorders_and_rescopes_by_relevance():
    candidates = [
        RetrievedClause(clause=make_clause("doc::a"), metadata=make_metadata(), score=0.5, source="hybrid"),
        RetrievedClause(clause=make_clause("doc::b"), metadata=make_metadata(), score=0.4, source="hybrid"),
    ]
    fake_client = MagicMock()
    fake_client.rerank.return_value = _FakeRerankResponse(
        [_FakeRerankResult(index=1, relevance_score=0.95), _FakeRerankResult(index=0, relevance_score=0.2)]
    )

    results = _reranker_with_fake_client(fake_client).rerank("some query", candidates, top_k=2)

    assert [r.clause.clause_id for r in results] == ["doc::b", "doc::a"]
    assert results[0].score == 0.95
    assert all(r.source == "hybrid+rerank" for r in results)


def test_rerank_of_empty_candidates_returns_empty_without_calling_client():
    fake_client = MagicMock()
    results = _reranker_with_fake_client(fake_client).rerank("query", [], top_k=5)

    assert results == []
    fake_client.rerank.assert_not_called()
