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

# Excluded from the "is this clause a match at all" overlap test only --
# BM25 scoring still sees every token (their low IDF already down-weights
# them there). Without this, any question containing "the" or "of"
# overlapped every clause, so the filter admitted the whole contract and
# top_k was filled with clauses that shared nothing but function words.
# Deliberately just English function words + question words and the
# ubiquitous contract modals ("shall", "will"): domain words that are merely
# frequent ("party", "agreement") stay, since a question about them is
# about something. "may" is kept out of the list (also a month).
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing done down during each either
    else etc ever every few for from further had has have having he her here hers herself him
    himself his how i if in into is it it's its itself just me might more most must my myself
    neither no nor not now of off on once only or other ought our ours ourselves out over own
    per please same she should so some such than that that's the their theirs them themselves
    then there there's these they this those through thus to too under until up upon us very
    was we were what what's when where whether which while who whom whose why will with within
    without would you your yours yourself yourselves shall hereby herein hereof hereto
    hereunder thereof therein thereto tell explain describe say says said
    """.split()
)


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
    by actual term overlap, not by score > 0 -- overlap on non-stopword
    terms (STOPWORDS)."""

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
        # A query of nothing but stopwords matches nothing here (vector
        # search still runs); it used to match everything.
        query_set = set(query_tokens) - STOPWORDS
        if not query_set:
            return []
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
