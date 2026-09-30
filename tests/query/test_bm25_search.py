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


def _law_and_fees_corpus():
    return [
        (make_clause("doc::law", text="The laws of the State of Delaware govern."), make_metadata()),
        (make_clause("doc::pay", text="The fees of the Supplier are due monthly."), make_metadata()),
        *_distractor_clauses(6),
    ]


def test_stopword_only_overlap_is_not_a_match():
    """L8: 'the' and 'of' appear in nearly every clause; before the fix any
    question containing them matched the whole contract."""
    index = InMemoryBM25Index(_law_and_fees_corpus())

    # The only tokens shared with any clause are stopwords ("law" != "laws").
    assert index.search("What is the governing law of the agreement?", top_k=10) == []


def test_content_word_still_matches_alongside_stopwords():
    index = InMemoryBM25Index(_law_and_fees_corpus())

    results = index.search("Which state is it, Delaware?", top_k=10)

    assert [r.clause.clause_id for r in results] == ["doc::law"]


def test_stopword_only_query_returns_nothing():
    corpus = [(make_clause("doc::a", text="The Supplier shall deliver."), make_metadata()), *_distractor_clauses(4)]
    assert InMemoryBM25Index(corpus).search("what is the", top_k=5) == []


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
