from query.filters.pii_sanitizer import detect_unstructured_pii, redact_structured_pii, sanitize_input


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
