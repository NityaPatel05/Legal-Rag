import pytest

from query.filters.pii_sanitizer import (
    detect_unstructured_pii,
    iban_valid,
    luhn_valid,
    redact_structured_pii,
    sanitize_input,
)


class FakeLLMClient:
    def __init__(self, pii_spans):
        self._pii_spans = pii_spans

    def generate_json(self, system_prompt, user_prompt):
        return {"pii_spans": self._pii_spans}


# --- redact_structured_pii ---


def test_redacts_ssn():
    result = redact_structured_pii("My SSN is 123-45-6789, please look into it.")
    assert "123-45-6789" not in result.sanitized_text
    assert "[REDACTED-SSN]" in result.sanitized_text
    assert result.any_pii_found is True
    assert result.findings[0].kind == "ssn"


def test_redacts_long_account_number():
    result = redact_structured_pii("The account number is 4111111111111111 on file.")
    assert "4111111111111111" not in result.sanitized_text
    assert "[REDACTED-NUMBER]" in result.sanitized_text


def test_no_pii_leaves_text_unchanged():
    text = "What does the termination clause say?"
    result = redact_structured_pii(text)
    assert result.sanitized_text == text
    assert result.any_pii_found is False


def test_short_numbers_are_not_redacted():
    """Ordinary legal numbers -- '30 days', 'Section 4' -- must not trip the
    account-number heuristic (9-19 consecutive digits)."""
    text = "Either party may terminate upon thirty (30) days notice per Section 4."
    result = redact_structured_pii(text)
    assert result.sanitized_text == text
    assert result.findings == []


# --- S5: more structured shapes, without redacting ordinary contract numbers ---


@pytest.mark.parametrize(
    ("text", "kind", "secret"),
    [
        ("SSN 123 45 6789 on file", "ssn", "123 45 6789"),
        ("SSN 123.45.6789 on file", "ssn", "123.45.6789"),
        ("card 4111 1111 1111 1111 exp 12/27", "card_number", "4111 1111 1111 1111"),
        ("card 4111-1111-1111-1111", "card_number", "4111-1111-1111-1111"),
        ("amex 3782 822463 10005", "card_number", "3782 822463 10005"),
        ("IBAN GB82 WEST 1234 5698 7654 32 please", "iban", "GB82 WEST 1234 5698 7654 32"),
        ("IBAN DE89370400440532013000.", "iban", "DE89370400440532013000"),
        ("notices to legal@acme-corp.com today", "email", "legal@acme-corp.com"),
        ("email j.doe+x@mail.example.co.uk", "email", "j.doe+x@mail.example.co.uk"),
        ("call (555) 123-4567", "phone", "(555) 123-4567"),
        ("call 555.123.4567", "phone", "555.123.4567"),
        ("call 555-123-4567", "phone", "555-123-4567"),
        ("call +1 555 123 4567", "phone", "+1 555 123 4567"),
        ("call +44 20 7946 0958", "phone", "+44 20 7946 0958"),
    ],
)
def test_redacts_structured_pii_shapes(text, kind, secret):
    result = redact_structured_pii(text)
    assert secret not in result.sanitized_text
    assert [f.kind for f in result.findings] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "Per Section 12.3.4 and Section 5.2(a), as amended on 2024-01-15 and 2024.01.15.",
        "Purchaser shall pay $1,000,000.00 (USD 2,500,000 in aggregate) within thirty (30) days.",
        "10,000 shares at 1.5% per month for 12 months; 100 000 EUR; version 1.2.3.4.",
        "Rule 10b-5; 15 U.S.C. § 78j(b); 26 U.S.C. 501(c)(3); ISO 9001:2015; Sections 101 through 105.",
        "Term: 2020-2025. Pages 100-200. Invoice No. 12345678; total 1,234,567,890.",
        # Separated 16 digits that fail Luhn: a reference number, not a card.
        "Purchase order PO 2024 0001 0002 0003 and ref 1234 5678 9012 3456.",
        # IBAN-shaped but the mod-97 check fails.
        "Reference GB00 WEST 1234 5698 7654 32.",
        # 3-2-4 spaced but not a valid SSN (area 000 / 9xx).
        "Codes 000 12 3456 and 912 34 5678.",
    ],
)
def test_ordinary_contract_numbers_are_not_redacted(text):
    result = redact_structured_pii(text)
    assert result.sanitized_text == text
    assert result.findings == []


def test_checksums():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB83 WEST 1234 5698 7654 32")


# --- detect_unstructured_pii ---


def test_detects_unstructured_pii_via_llm():
    llm_client = FakeLLMClient([{"text": "John Smith", "kind": "name"}])
    findings = detect_unstructured_pii("Please contact John Smith about this.", llm_client=llm_client)

    assert len(findings) == 1
    assert findings[0].kind == "name"
    assert findings[0].detection_method == "llm"


def test_detect_unstructured_pii_empty_spans_returns_empty_list():
    llm_client = FakeLLMClient([])
    assert detect_unstructured_pii("Nothing sensitive here.", llm_client=llm_client) == []


# --- sanitize_input (composition) ---


def test_sanitize_input_without_llm_client_only_does_structured_redaction():
    result = sanitize_input("SSN 123-45-6789, contact John Smith.", llm_client=None)
    assert "[REDACTED-SSN]" in result.sanitized_text
    assert "John Smith" in result.sanitized_text  # unstructured detection never ran


def test_sanitize_input_with_llm_client_redacts_both():
    llm_client = FakeLLMClient([{"text": "John Smith", "kind": "name"}])
    result = sanitize_input("SSN 123-45-6789, contact John Smith.", llm_client=llm_client)

    assert "[REDACTED-SSN]" in result.sanitized_text
    assert "John Smith" not in result.sanitized_text
    assert "[REDACTED-NAME]" in result.sanitized_text
    kinds = {f.kind for f in result.findings}
    assert kinds == {"ssn", "name"}
