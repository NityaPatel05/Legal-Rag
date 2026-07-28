"""In-memory BM25 index over a corpus of clauses — the keyword-search half of
CLAUDE.md's hybrid-retrieval decision (vector + BM25). BM25Index is the seam
hybrid.py depends on; InMemoryBM25Index is the real implementation
(rank_bm25, lazy-imported).

SECURITY NOTE: unlike vector search, this index has no per-query permission
filter — a BM25Okapi instance is built once over whatever corpus it's given.
The caller MUST construct one InMemoryBM25Index per tenant/matter scope (or
otherwise pre-filter the corpus) rather than share one index across tenants;
hybrid.py additionally drops any mismatched result before fusion as
defense-in-depth, but that's a second layer, not a substitute for a
correctly-scoped corpus."""

from __future__ import annotations

from typing import Protocol

from ingestion.schemas import Clause, ClauseMetadata
from query.schemas import RetrievedClause


class BM25Index(Protocol):
    def search(self, query_text: str, *, top_k: int) -> list[RetrievedClause]: ...


class InMemoryBM25Index:
    def __init__(self, corpus: list[tuple[Clause, ClauseMetadata]]):
        self._corpus = corpus
        self._bm25 = None

    def _bm25_instance(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi  # lazy import

            tokenized = [clause.text.lower().split() for clause, _ in self._corpus]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
        return self._bm25

    def search(self, query_text: str, *, top_k: int) -> list[RetrievedClause]:
        if not self._corpus:
            return []
        bm25 = self._bm25_instance()
        scores = bm25.get_scores(query_text.lower().split())
        ranked = sorted(zip(self._corpus, scores), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedClause(clause=clause, metadata=metadata, score=float(score), source="bm25")
            for (clause, metadata), score in ranked[:top_k]
            if score > 0
        ]
