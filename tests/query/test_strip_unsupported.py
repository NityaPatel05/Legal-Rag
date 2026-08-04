import pytest

from query.validation.citation_validator import Citation, CitationValidationResult
from query.validation.strip_unsupported import Claim, strip_unsupported_claims


def _result(is_valid: bool) -> CitationValidationResult:
    return CitationValidationResult(
        citation=Citation(section_id="1", claim_text="x"),
        exists=is_valid,
        in_context=is_valid,
        span_supported=is_valid,
    )


def test_valid_claims_are_kept_and_invalid_ones_stripped():
    claims = [
        Claim(claim_text="Governed by Delaware law.", citation_section_id="4"),
        Claim(claim_text="Ninety days notice required.", citation_section_id="3"),
    ]
    results = [_result(is_valid=True), _result(is_valid=False)]

    outcome = strip_unsupported_claims(claims, results)

    assert [c.claim_text for c in outcome.kept] == ["Governed by Delaware law."]
    assert [c.claim_text for c in outcome.stripped] == ["Ninety days notice required."]
    assert outcome.any_stripped is True


def test_all_valid_means_nothing_stripped():
    claims = [Claim(claim_text="a", citation_section_id="1")]
    outcome = strip_unsupported_claims(claims, [_result(is_valid=True)])

    assert outcome.stripped == []
    assert outcome.any_stripped is False


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        strip_unsupported_claims(
            [Claim(claim_text="a", citation_section_id="1")],
            [],
        )
