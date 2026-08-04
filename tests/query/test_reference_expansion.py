from ingestion.schemas import SectionReference
from query.retrieval.reference_expansion import expand_with_references
from query.schemas import RetrievedClause
from tests.query.factories import make_clause, make_metadata


def test_expands_referenced_clause_not_already_included():
    c1 = make_clause("doc::c1", section_id="1")
    c2 = make_clause("doc::c2", section_id="2")
    top = [RetrievedClause(clause=c1, metadata=make_metadata(), score=1.0, source="hybrid")]
    references = [SectionReference(source_clause_id="doc::c1", referenced_section_id="2", raw_text="Section 2")]

    expanded = expand_with_references(
        top, all_clauses_by_section_id={"2": (c2, make_metadata())}, references=references
    )

    ids = [rc.clause.clause_id for rc in expanded]
    assert ids == ["doc::c1", "doc::c2"]
    assert expanded[1].source == "reference_expansion"


def test_does_not_duplicate_an_already_included_clause():
    c1 = make_clause("doc::c1", section_id="1")
    c2 = make_clause("doc::c2", section_id="2")
    top = [
        RetrievedClause(clause=c1, metadata=make_metadata(), score=1.0, source="hybrid"),
        RetrievedClause(clause=c2, metadata=make_metadata(), score=0.9, source="hybrid"),
    ]
    references = [SectionReference(source_clause_id="doc::c1", referenced_section_id="2", raw_text="Section 2")]

    expanded = expand_with_references(
        top, all_clauses_by_section_id={"2": (c2, make_metadata())}, references=references
    )
    assert len(expanded) == 2


def test_only_expands_one_hop_from_already_retrieved_clauses():
    """A reference whose source clause isn't already retrieved shouldn't
    pull anything in -- one hop out, not the whole reference graph."""
    top = [RetrievedClause(clause=make_clause("doc::c1", section_id="1"), metadata=make_metadata(), score=1.0, source="hybrid")]
    c3 = make_clause("doc::c3", section_id="3")
    references = [SectionReference(source_clause_id="doc::c99", referenced_section_id="3", raw_text="Section 3")]

    expanded = expand_with_references(
        top, all_clauses_by_section_id={"3": (c3, make_metadata())}, references=references
    )
    assert len(expanded) == 1


def test_retrieved_clauses_default_to_the_trust_label():
    rc = RetrievedClause(clause=make_clause("doc::c1"), metadata=make_metadata(), score=1.0, source="vector")
    assert "evidence only" in rc.trust_label
