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

import re
from typing import Protocol

from ingestion.schemas import Clause, ClauseMetadata
from query.schemas import RetrievedClause

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping "2.1" and "party's" intact.
    Whitespace-splitting (the original) kept punctuation attached, so
    "Delaware." in a clause never matched "Delaware" in a query."""
    return _TOKEN.findall(text.lower())


class BM25Index(Protocol):
    def search(self, query_text: str, *, top_k: int) -> list[RetrievedClause]: ...


class InMemoryBM25Index:
    """BM25Plus, not BM25Okapi. The corpus here is ONE contract's clauses, so
    it's small, and Okapi's IDF is <= 0 for any term in half or more of the
    documents: a 2-clause contract where "Delaware" appears in one clause
    scored every clause 0, and the `score > 0` filter then dropped the
    match. BM25Plus's IDF is always positive. It also adds a small positive
    score to documents with NO query terms, so "is this a match" is decided
    by actual term overlap, not by score > 0."""

    def __init__(self, corpus: list[tuple[Clause, ClauseMetadata]]):
        self._corpus = corpus
        self._bm25 = None
        self._token_sets: list[set[str]] = []

    def _bm25_instance(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Plus  # lazy import

            # Heading is indexed with the text: a passage split from the
            # middle of a long clause doesn't contain its own heading.
            tokenized = [tokenize(f"{clause.heading or ''} {clause.text}") for clause, _ in self._corpus]
            self._token_sets = [set(tokens) for tokens in tokenized]
            self._bm25 = BM25Plus(tokenized) if tokenized else None
        return self._bm25

    def search(self, query_text: str, *, top_k: int) -> list[RetrievedClause]:
        if not self._corpus:
            return []
        bm25 = self._bm25_instance()
        query_tokens = tokenize(query_text)
        query_set = set(query_tokens)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (entry, score)
                for entry, score, tokens in zip(self._corpus, scores, self._token_sets)
                if tokens & query_set
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            RetrievedClause(clause=clause, metadata=metadata, score=float(score), source="bm25")
            for (clause, metadata), score in ranked[:top_k]
        ]
