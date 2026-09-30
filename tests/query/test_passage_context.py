"""R4: small-to-big -- retrieved passages are widened with neighbours from
the same parent clause, bounded, never crossing a clause boundary."""

from ingestion.schemas import Clause
from query.retrieval.passage_context import expand_with_sibling_passages
from query.schemas import RetrievedClause
from tests.query.factories import make_clause, make_metadata


def _passage(parent: str, index: int, start: int) -> Clause:
    return Clause(
        clause_id=f"{parent}.p{index:02d}",
        section_id="5",
        text=f"passage {index} text",
        start_offset=start,
        end_offset=start + 10,
        parent_clause_id=parent,
    )


def _by_id(clauses):
    return {c.clause_id: (c, make_metadata()) for c in clauses}


def _hit(clause, score=1.0):
    return RetrievedClause(clause=clause, metadata=make_metadata(), score=score, source="hybrid")


def test_neighbours_inserted_around_hit_in_document_order():
    passages = [_passage("doc::clause-005", i, i * 10) for i in range(5)]
    result = expand_with_sibling_passages([_hit(passages[2])], all_clauses_by_id=_by_id(passages))

    assert [r.clause.clause_id for r in result] == [
        "doc::clause-005.p01",
        "doc::clause-005.p02",
        "doc::clause-005.p03",
    ]
    assert [r.source for r in result] == ["passage_context", "hybrid", "passage_context"]


def test_never_crosses_into_another_clause():
    passages = [_passage("doc::clause-005", 0, 0), _passage("doc::clause-005", 1, 10)]
    other = [_passage("doc::clause-006", 0, 20)]
    unsplit = make_clause("doc::clause-007", start=30)
    result = expand_with_sibling_passages(
        [_hit(passages[1]), _hit(unsplit)], all_clauses_by_id=_by_id(passages + other + [unsplit])
    )

    ids = [r.clause.clause_id for r in result]
    assert "doc::clause-006.p00" not in ids
    assert ids == ["doc::clause-005.p00", "doc::clause-005.p01", "doc::clause-007"]


def test_budget_is_bounded_and_best_hit_claims_it_first():
    a = [_passage("doc::clause-001", i, i * 10) for i in range(3)]
    b = [_passage("doc::clause-002", i, 100 + i * 10) for i in range(3)]
    result = expand_with_sibling_passages(
        [_hit(a[1], score=0.1), _hit(b[1], score=0.9)],
        all_clauses_by_id=_by_id(a + b),
        max_added=2,
    )

    added = [r.clause.clause_id for r in result if r.source == "passage_context"]
    assert added == ["doc::clause-002.p00", "doc::clause-002.p02"]


def test_no_duplicate_when_neighbour_already_retrieved():
    passages = [_passage("doc::clause-005", i, i * 10) for i in range(3)]
    result = expand_with_sibling_passages(
        [_hit(passages[0]), _hit(passages[1])], all_clauses_by_id=_by_id(passages)
    )
    ids = [r.clause.clause_id for r in result]
    assert len(ids) == len(set(ids)) == 3


def test_passthrough_without_document_lookup():
    hit = _hit(make_clause("doc::c1"))
    assert expand_with_sibling_passages([hit], all_clauses_by_id=None) == [hit]
