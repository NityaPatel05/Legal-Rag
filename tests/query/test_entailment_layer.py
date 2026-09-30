"""R6: optional entailment layer after the verbatim-quote checks."""

from query.schemas import RetrievedClause
from query.validation.citation_validator import Citation, validate_citation
from tests.query.factories import make_clause, make_metadata

CLAUSE = make_clause(
    "doc::clause-004",
    text="The Supplier shall indemnify the Customer against all third-party claims.",
    section_id="4",
)
SHOWN = [RetrievedClause(clause=CLAUSE, metadata=make_metadata(), score=1.0, source="hybrid")]
# Same words, same numbers, same negation -- the obligor is swapped. Every
# lexical check passes this; only entailment catches it.
SWAPPED = Citation(
    section_id="doc::clause-004",
    claim_text="The Customer shall indemnify the Supplier against all third-party claims.",
    supporting_quote="The Supplier shall indemnify the Customer against all third-party claims",
)


class FakeNLI:
    """Entails only when the hypothesis names the same obligor first."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def entails(self, premise: str, hypothesis: str) -> bool:
        self.calls.append((premise, hypothesis))
        return premise.lower().split()[:2] == hypothesis.lower().split()[:2]


def test_lexical_checks_alone_accept_a_swapped_obligor():
    assert validate_citation(SWAPPED, source_clauses=[CLAUSE], shown_clauses=SHOWN).is_valid


def test_entailment_layer_rejects_swapped_obligor():
    result = validate_citation(
        SWAPPED, source_clauses=[CLAUSE], shown_clauses=SHOWN, entailment_checker=FakeNLI()
    )
    assert not result.is_valid
    assert "entailed" in result.failure_reason


def test_entailment_layer_keeps_a_correct_claim():
    correct = SWAPPED.model_copy(
        update={"claim_text": "The Supplier shall indemnify the Customer against third-party claims."}
    )
    checker = FakeNLI()
    assert validate_citation(
        correct, source_clauses=[CLAUSE], shown_clauses=SHOWN, entailment_checker=checker
    ).is_valid
    # Premise is the quote's enclosing sentence, not the whole document.
    assert checker.calls[0][0].startswith("the supplier shall indemnify")


def test_entailment_not_called_when_a_string_check_already_failed():
    wrong_quote = SWAPPED.model_copy(update={"supporting_quote": "text that is not in the clause at all"})
    checker = FakeNLI()
    validate_citation(wrong_quote, source_clauses=[CLAUSE], shown_clauses=SHOWN, entailment_checker=checker)
    assert checker.calls == []
