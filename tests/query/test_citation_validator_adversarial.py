"""Adversarial table for the citation validator's claim-level checks
(numbers B4, repeated quotes B5, modality/parties/negation B6), both
directions: a wrong claim that used to pass must fail, and a correct claim
that used to fail (or might newly fail) must pass. A false acceptance is a
hallucination reaching a lawyer; a false rejection strips a correct claim --
both are asserted here."""

import sys
import threading
import types

import pytest

import query.validation.citation_validator as citation_validator
from query.schemas import RetrievedClause
from query.validation.citation_validator import Citation, CrossEncoderEntailmentChecker, validate_citation
from tests.query.factories import make_clause, make_metadata

FIXTURE_TERMINATION = (
    "Either party may terminate this Agreement for material breach on thirty (30) days' written notice, "
    "and termination does not affect the Initial Term obligations accrued under Section 2.1."
)
ASSIGNMENT = "Licensee may not assign this Agreement, except to an Affiliate, without consent."
FEES = "Fees are payable within thirty (30) days of invoice. Disputed fees are payable within ninety (90) days of resolution."
INDEMNITY = "The Supplier shall indemnify the Customer against all third-party claims."
TERMINATION_30 = "Licensee may terminate this Agreement upon thirty (30) days written notice."

# (clause text, claim, supporting quote, expected valid)
CASES = [
    # --- B4: numbers -- must fail -------------------------------------------------
    pytest.param(
        "Licensee's aggregate liability is capped at $1 billion in any year.",
        "Licensee's aggregate liability is capped at $1 million in any year.",
        "aggregate liability is capped at $1 billion",
        False, id="b4-million-vs-billion",
    ),
    pytest.param(
        "The late fee is 5.2% per month on overdue amounts.",
        "The late fee is 2.5% per month on overdue amounts.",
        "The late fee is 5.2% per month",
        False, id="b4-decimal-digit-order",
    ),
    pytest.param(
        "Licensee shall pay within one hundred days of invoice.",
        "Licensee shall pay within one hundred twenty (120) days of invoice.",
        "Licensee shall pay within one hundred days",
        False, id="b4-compound-number-words",
    ),
    pytest.param(
        "Licensee shall pay within one hundred twenty (120) days of invoice.",
        "Licensee shall pay within twenty days of invoice.",
        "Licensee shall pay within one hundred twenty (120) days",
        False, id="b4-twenty-inside-one-hundred-twenty",
    ),
    pytest.param(
        "The late fee is 1.5% per month on overdue amounts.",
        "The late fee is 5% per month on overdue amounts.",
        "The late fee is 1.5% per month",
        False, id="b4-5-vs-1.5-percent",
    ),
    pytest.param(
        "The warranty period is 30 days from delivery.",
        "The warranty period is 30 months from delivery.",
        "The warranty period is 30 days from delivery",
        False, id="b4-unit-months-vs-days",
    ),
    pytest.param(
        "Supplier shall respond within ten (10) business days of a request.",
        "Supplier shall respond within ten (10) days of a request.",
        "respond within ten (10) business days",
        False, id="b4-business-days-vs-days",
    ),
    pytest.param(
        "Licensee may terminate this Agreement upon ninety (90) days written notice.",
        "Licensee may terminate this Agreement upon thirty (30) days written notice.",
        "terminate this Agreement upon ninety (90) days",
        False, id="b4-regression-thirty-vs-ninety",
    ),
    pytest.param(
        TERMINATION_30,
        "Licensee may terminate this Agreement upon ninety days written notice.",
        "terminate this Agreement upon thirty (30) days",
        False, id="b4-regression-number-word-vs-digit",
    ),
    # --- B4: numbers -- must pass -------------------------------------------------
    pytest.param(
        TERMINATION_30,
        "Under Section 12.2, Licensee may terminate this Agreement upon thirty (30) days written notice.",
        "Licensee may terminate this Agreement upon thirty (30) days",
        True, id="b4-section-number-is-not-a-quantity",
    ),
    pytest.param(
        TERMINATION_30,
        "Pursuant to Sections 5.2 and 7, Licensee may terminate this Agreement upon thirty (30) days written notice.",
        "Licensee may terminate this Agreement upon thirty (30) days",
        True, id="b4-section-list-is-not-a-quantity",
    ),
    pytest.param(
        "Licensee's aggregate liability is capped at $1,000,000 in any year.",
        "Licensee's aggregate liability is capped at $1 million in any year.",
        "aggregate liability is capped at $1,000,000",
        True, id="b4-million-equals-1000000",
    ),
    pytest.param(
        "Licensee's aggregate liability is capped at one million U.S. dollars in any year.",
        "Licensee's aggregate liability is capped at $1,000,000 in any year.",
        "aggregate liability is capped at one million U.S. dollars",
        True, id="b4-number-words-with-currency",
    ),
    pytest.param(
        "The late fee is five percent (5%) per month on overdue amounts.",
        "The late fee is 5% per month on overdue amounts.",
        "The late fee is five percent (5%) per month",
        True, id="b4-percent-word-and-sign",
    ),
    pytest.param(
        TERMINATION_30,
        "Licensee may terminate this Agreement upon a 30-day written notice.",
        "Licensee may terminate this Agreement upon thirty (30) days",
        True, id="b4-hyphenated-unit",
    ),
    pytest.param(
        "Licensee shall pay within one hundred twenty (120) days of invoice.",
        "Licensee shall pay within 120 days of invoice.",
        "Licensee shall pay within one hundred twenty (120) days",
        True, id="b4-compound-words-equal-digits",
    ),
    # --- B5: repeated quote ---------------------------------------------------------
    pytest.param(
        FEES,
        "Disputed fees are payable within thirty (30) days of invoice.",
        "fees are payable within",
        False, id="b5-duplicate-quote-lets-false-claim-borrow-wrong-sentence",
    ),
    pytest.param(
        FEES,
        "Disputed fees are payable within ninety (90) days of resolution.",
        "fees are payable within",
        False, id="b5-duplicate-quote-is-ambiguous-even-for-true-claim",
    ),
    pytest.param(
        FEES,
        "Disputed fees are payable within ninety (90) days of resolution.",
        "Disputed fees are payable within",
        True, id="b5-unique-quote-true-claim",
    ),
    pytest.param(
        "The Supplier shall deliver the goods and the Supplier shall invoice the goods monthly.",
        "The Supplier shall invoice the goods monthly.",
        "the Supplier shall",
        True, id="b5-repeat-within-one-sentence-is-fine",
    ),
    # --- B6: modality ----------------------------------------------------------------
    pytest.param(
        "Licensee may audit the Supplier's records once per year.",
        "Licensee must audit the Supplier's records once per year.",
        "Licensee may audit the Supplier's records",
        False, id="b6-may-to-must",
    ),
    pytest.param(
        "Licensee shall maintain insurance coverage during the Term.",
        "Licensee may maintain insurance coverage during the Term.",
        "Licensee shall maintain insurance coverage",
        False, id="b6-shall-to-may",
    ),
    pytest.param(
        "Licensee is not required to purchase any minimum quantity.",
        "Licensee is required to purchase a minimum quantity.",
        "Licensee is not required to purchase any minimum quantity",
        False, id="b6-no-obligation-to-obligation",
    ),
    pytest.param(
        "Licensee shall maintain insurance coverage during the Term.",
        "Licensee will maintain insurance coverage during the Term.",
        "Licensee shall maintain insurance coverage",
        True, id="b6-will-equals-shall",
    ),
    pytest.param(
        "Neither party shall assign this Agreement without the other party's consent.",
        "No party may assign this Agreement without the other party's consent.",
        "Neither party shall assign this Agreement",
        True, id="b6-no-party-may-equals-neither-shall",
    ),
    pytest.param(
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "The Agreement is governed by Delaware law.",
        "governed by the laws of the State of Delaware",
        True, id="b6-regression-declarative-claim-from-modal-evidence",
    ),
    # --- B6: parties -----------------------------------------------------------------
    pytest.param(
        "Licensee shall pay all fees within thirty (30) days. The Company shall deliver the Software.",
        "The Company shall pay all fees within thirty (30) days.",
        "shall pay all fees within thirty (30) days",
        False, id="b6-party-from-another-sentence",
    ),
    pytest.param(
        ASSIGNMENT,
        "Company may assign this Agreement to an Affiliate.",
        "except to an Affiliate",
        False, id="b6-party-not-in-clause",
    ),
    pytest.param(
        INDEMNITY,
        "The Customer shall indemnify the Supplier against all third-party claims.",
        "The Supplier shall indemnify the Customer",
        False, id="b6-obligor-swap",
    ),
    pytest.param(
        "Licensee shall pay the fees, and Licensor shall deliver the Software.",
        "Licensor shall pay the fees.",
        "pay the fees",
        False, id="b6-obligor-from-other-segment",
    ),
    pytest.param(
        INDEMNITY,
        "The Customer shall be indemnified by the Supplier against all third-party claims.",
        "The Supplier shall indemnify the Customer",
        True, id="b6-passive-restatement",
    ),
    pytest.param(
        "Either Party may terminate this Agreement upon thirty (30) days notice.",
        "Licensee may terminate this Agreement upon thirty (30) days notice.",
        "may terminate this Agreement upon thirty (30) days",
        True, id="b6-either-party-covers-a-named-party",
    ),
    pytest.param(
        "Licensee shall keep accurate records; and shall provide copies to Licensor on request.",
        "Licensee shall provide copies of records to Licensor on request.",
        "shall provide copies to Licensor on request",
        True, id="b6-subject-before-semicolon",
    ),
    pytest.param(
        "This Agreement is between Acme Corp and Globex LLC.",
        "The Agreement is between Acme Corp and Globex LLC.",
        "between Acme Corp and Globex LLC",
        True, id="b6-regression-named-parties",
    ),
    # --- B6: negation ----------------------------------------------------------------
    pytest.param(
        ASSIGNMENT,
        "Licensee may assign this Agreement to an Affiliate.",
        "Licensee may not assign this Agreement, except to an Affiliate",
        True, id="b6-carve-out-positive-claim",
    ),
    pytest.param(
        ASSIGNMENT,
        "Licensee may assign this Agreement to any third party.",
        "Licensee may not assign this Agreement, except to an Affiliate",
        False, id="b6-carve-out-does-not-cover-other-assignees",
    ),
    pytest.param(
        ASSIGNMENT,
        "Licensee may not assign this Agreement to an Affiliate.",
        "Licensee may not assign this Agreement, except to an Affiliate",
        False, id="b6-negating-the-carve-out",
    ),
    pytest.param(
        "Fees are not refundable unless the Agreement is terminated for cause.",
        "Fees are refundable if the Agreement is terminated for cause.",
        "Fees are not refundable unless the Agreement is terminated for cause",
        True, id="b6-unless-carve-out",
    ),
    pytest.param(
        "Fees are not refundable unless the Agreement is terminated for cause.",
        "Fees are refundable.",
        "Fees are not refundable unless the Agreement is terminated for cause",
        False, id="b6-unless-dropped",
    ),
    pytest.param(
        FIXTURE_TERMINATION,
        "Either party may terminate this Agreement for material breach on thirty (30) days' written notice.",
        "terminate this Agreement for material breach",
        True, id="b6-l19-negation-in-other-segment",
    ),
    pytest.param(
        FIXTURE_TERMINATION,
        "Either party may not terminate this Agreement for material breach.",
        "terminate this Agreement for material breach",
        False, id="b6-l19-inverse-added-not",
    ),
    pytest.param(
        FIXTURE_TERMINATION,
        "Termination does not affect the Initial Term obligations accrued under Section 2.1.",
        "termination does not affect the Initial Term obligations",
        True, id="b6-l19-second-segment",
    ),
    pytest.param(
        "Fees paid are not refundable.",
        "Fees paid are refundable.",
        "Fees paid are not refundable",
        False, id="b6-dropped-not-inside-quote",
    ),
    pytest.param(
        "Fees are not refundable, provided that Customer may request a credit.",
        "Fees are refundable, provided that Customer may request a credit.",
        "provided that Customer may request a credit",
        False, id="b6-dropped-not-just-outside-quote-segment",
    ),
    pytest.param(
        "Licensee shall not copy the Software, and distribute copies to third parties.",
        "Licensee may distribute copies to third parties.",
        "distribute copies to third parties",
        False, id="b6-verbless-and-part-keeps-its-not",
    ),
    pytest.param(
        "Licensee may not assign this Agreement, but Licensee may assign its rights to an Affiliate with consent.",
        "Licensee may assign its rights to an Affiliate with consent.",
        "Licensee may assign its rights to an Affiliate with consent",
        True, id="b6-but-segment",
    ),
    pytest.param(
        "Customer shall pay each invoice not later than thirty (30) days after receipt.",
        "Customer shall pay each invoice no later than thirty (30) days after receipt.",
        "pay each invoice not later than thirty (30) days",
        True, id="b6-no-later-than-equals-not-later-than",
    ),
    pytest.param(
        "Customer shall pay each invoice not later than thirty (30) days after receipt.",
        "Customer shall pay each invoice within thirty (30) days after receipt.",
        "pay each invoice not later than thirty (30) days",
        True, id="b6-deadline-is-not-a-negation",
    ),
    pytest.param(
        "The Initial Term shall be not less than three (3) years.",
        "The Initial Term shall be at least three (3) years.",
        "not less than three (3) years",
        True, id="b6-at-least-equals-not-less-than",
    ),
    pytest.param(
        "Licensee shall give no more than ten (10) days notice.",
        "Licensee shall give more than ten (10) days notice.",
        "no more than ten (10) days notice",
        False, id="b6-dropped-no-more-than",
    ),
]


def _shown(clause):
    return [RetrievedClause(clause=clause, metadata=make_metadata(), score=1.0, source="hybrid")]


@pytest.mark.parametrize(("clause_text", "claim", "quote", "expected_valid"), CASES)
def test_adversarial_claims(clause_text, claim, quote, expected_valid):
    clause = make_clause("doc::clause-001", text=clause_text, section_id="1")
    result = validate_citation(
        Citation(section_id=clause.clause_id, claim_text=claim, supporting_quote=quote),
        source_clauses=[clause],
        shown_clauses=_shown(clause),
    )
    assert result.is_valid is expected_valid, result.failure_reason


def test_ambiguous_quote_reason_asks_for_more_context():
    clause = make_clause("doc::clause-001", text=FEES)
    result = validate_citation(
        Citation(
            section_id=clause.clause_id,
            claim_text="Disputed fees are payable within thirty (30) days of invoice.",
            supporting_quote="fees are payable within",
        ),
        source_clauses=[clause],
        shown_clauses=_shown(clause),
    )
    assert "ambiguous" in result.failure_reason and "2 times" in result.failure_reason


def test_number_failure_names_the_missing_quantity():
    clause = make_clause("doc::clause-001", text="The warranty period is 30 days from delivery.")
    result = validate_citation(
        Citation(
            section_id=clause.clause_id,
            claim_text="The warranty period is 30 months from delivery.",
            supporting_quote="The warranty period is 30 days",
        ),
        source_clauses=[clause],
        shown_clauses=_shown(clause),
    )
    assert "30 month" in result.failure_reason


def test_legacy_no_quote_path_still_checks_the_whole_clause():
    clause = make_clause("doc::clause-001", text="Licensee may audit the records. Fees are not refundable.")
    wrong = Citation(section_id=clause.clause_id, claim_text="Licensee must audit the records.")
    assert not validate_citation(wrong, source_clauses=[clause], shown_clauses=_shown(clause)).is_valid


# --- L11: dead helpers removed ------------------------------------------------------


def test_dead_helpers_are_gone():
    assert not hasattr(citation_validator, "check_exists")
    assert not hasattr(citation_validator, "check_in_context")


# --- L4: CrossEncoderEntailmentChecker loading -----------------------------------------


def _fake_sentence_transformers(monkeypatch, id2label, constructed):
    class FakeCrossEncoder:
        def __init__(self, name):
            constructed.append(name)
            self.model = types.SimpleNamespace(config=types.SimpleNamespace(id2label=id2label))

        def predict(self, pairs, apply_softmax):
            return [[0.1, 0.8, 0.1]]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))


def test_checker_without_entailment_label_raises_a_clear_error(monkeypatch):
    _fake_sentence_transformers(monkeypatch, {0: "LABEL_0", 1: "LABEL_1"}, [])
    checker = CrossEncoderEntailmentChecker("some/model")
    with pytest.raises(ValueError, match="no entailment label.*LABEL_0"):
        checker.entails("premise", "hypothesis")


def test_checker_reads_entailment_index_and_loads_once_under_concurrency(monkeypatch):
    constructed: list[str] = []
    _fake_sentence_transformers(monkeypatch, {0: "contradiction", 1: "entailment", 2: "neutral"}, constructed)
    checker = CrossEncoderEntailmentChecker("some/model")
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(checker.entails("p", "h"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True] * 8
    assert constructed == ["some/model"]
