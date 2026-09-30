"""R2: golden-set loader, scoring, and the certification gate."""

from pathlib import Path

import pytest

from eval.golden_set.golden import (
    MIN_CERTIFIABLE_ITEMS,
    GoldenItem,
    evaluate,
    is_certifiable,
    load_golden_set,
)
from ingestion.parsing.clause_parser import parse_clauses

EXAMPLE = Path("eval/golden_set/golden_set.example.jsonl")
FIXTURE = Path("tests/ingestion/fixtures/sample_contract.txt")


def _item(i=0, **overrides):
    base = dict(
        id=f"q-{i}",
        tenant_id="t",
        matter_id="m",
        document_id="d",
        question="q?",
        acceptable_clause_ids=["d::clause-001"],
        reviewed_by="A. Lawyer",
    )
    return GoldenItem(**{**base, **overrides})


def _answer(clause_id="d::clause-001", status="answered", text="Delaware law applies."):
    return {"final_status": status, "answer_text": text, "citations": [{"clause_id": clause_id}]}


def test_example_set_loads_and_refers_to_real_fixture_clauses():
    items = load_golden_set(EXAMPLE)
    clause_ids = {c.clause_id for c in parse_clauses(FIXTURE.read_text(encoding="utf-8"), document_id="sample-msa")}
    assert items
    for item in items:
        assert set(item.acceptable_clause_ids) <= clause_ids, item.id


def test_example_set_is_not_certifiable():
    assert not is_certifiable(load_golden_set(EXAMPLE))


def test_certification_needs_enough_reviewed_items():
    reviewed = [_item(i) for i in range(MIN_CERTIFIABLE_ITEMS)]
    assert is_certifiable(reviewed)
    assert not is_certifiable(reviewed[:-1])
    assert not is_certifiable(reviewed[:-1] + [_item(999, reviewed_by=None)])


def test_one_wrong_citation_fails_the_default_gate():
    items = [_item(i) for i in range(MIN_CERTIFIABLE_ITEMS)]
    answers = {item.id: _answer() for item in items}
    answers["q-0"] = _answer(clause_id="d::clause-999")

    report = evaluate(items, lambda item: answers[item.id])
    assert report.wrong_citations == 1
    assert not report.passed

    assert evaluate(items, lambda item: _answer()).passed


def test_refusal_and_answer_coverage():
    refuse = _item(1, expect_refusal=True, acceptable_clause_ids=[])
    must = _item(2, answer_must_contain=["delaware"])
    report = evaluate(
        [refuse, must],
        lambda item: {"final_status": "flagged_for_review", "citations": []}
        if item.id == refuse.id
        else _answer(),
    )
    assert report.refusal_accuracy == 1.0
    assert report.answer_coverage == 1.0
    assert report.citation_accuracy == 1.0


def test_run_that_cites_nothing_does_not_pass():
    items = [_item(i) for i in range(MIN_CERTIFIABLE_ITEMS)]
    report = evaluate(items, lambda item: {"final_status": "flagged_for_review", "citations": []})
    assert report.citation_accuracy is None
    assert not report.passed


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "g.jsonl"
    line = _item().model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(path)
