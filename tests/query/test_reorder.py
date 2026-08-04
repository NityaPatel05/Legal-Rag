from query.retrieval.reorder import reorder_for_generation
from query.schemas import RetrievedClause
from tests.query.factories import make_clause, make_metadata


def _rc(clause_id: str, score: float) -> RetrievedClause:
    return RetrievedClause(clause=make_clause(clause_id), metadata=make_metadata(), score=score, source="hybrid")


def test_best_clause_is_first_and_second_best_is_last():
    clauses = [_rc("a", 0.9), _rc("b", 0.7), _rc("c", 0.5), _rc("d", 0.3), _rc("e", 0.1)]
    reordered = reorder_for_generation(clauses)

    ids = [rc.clause.clause_id for rc in reordered]
    assert ids[0] == "a"
    assert ids[-1] == "b"
    assert set(ids) == {"a", "b", "c", "d", "e"}


def test_input_order_does_not_matter_only_score_does():
    shuffled = [_rc("c", 0.5), _rc("a", 0.9), _rc("e", 0.1), _rc("b", 0.7), _rc("d", 0.3)]
    ids = [rc.clause.clause_id for rc in reorder_for_generation(shuffled)]
    assert ids[0] == "a"
    assert ids[-1] == "b"


def test_lists_of_two_or_fewer_are_returned_unchanged():
    clauses = [_rc("a", 0.9), _rc("b", 0.1)]
    assert [rc.clause.clause_id for rc in reorder_for_generation(clauses)] == ["a", "b"]


def test_empty_list_returns_empty():
    assert reorder_for_generation([]) == []
