from query.retrieval.bm25_search import InMemoryBM25Index
from tests.query.factories import make_clause, make_metadata


def _distractor_clauses(n: int, start: int = 0):
    """Filler docs on an unrelated topic, so the term(s) under test have a
    document frequency well below corpus size — with too few distractors,
    BM25's IDF degenerates to zero (a term in exactly half a tiny corpus has
    no discriminative weight) or negative (a term in every document), and a
    correctly-implemented score>0 filter would drop everything."""
    return [
        (make_clause(f"doc::filler{i}", text="This clause covers payment terms and invoicing schedules."), make_metadata())
        for i in range(start, start + n)
    ]


def test_returns_the_more_relevant_clause_first():
    corpus = [
        (make_clause("doc::c1", text="This clause covers governing law and jurisdiction."), make_metadata()),
        *_distractor_clauses(8),
    ]
    index = InMemoryBM25Index(corpus)

    results = index.search("governing law", top_k=5)

    assert len(results) == 1  # only c1 has any keyword overlap
    assert results[0].clause.clause_id == "doc::c1"
    assert results[0].source == "bm25"


def test_empty_corpus_returns_no_results():
    index = InMemoryBM25Index([])
    assert index.search("anything", top_k=5) == []


def test_top_k_limits_results():
    corpus = [
        (make_clause(f"doc::c{i}", text="termination termination termination"), make_metadata())
        for i in range(5)
    ] + _distractor_clauses(10, start=5)
    index = InMemoryBM25Index(corpus)

    results = index.search("termination", top_k=3)
    assert len(results) == 3
