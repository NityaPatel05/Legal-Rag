"""Input sanitization guardrail — CLAUDE.md #8: regex for structured PII
(SSNs, account numbers) + prompted-LLM detection for unstructured PII
(names, addresses, signatures). Redact before sending to any third-party
model provider.

Split into two independent pieces on purpose:

- `redact_structured_pii` is regex-only, free, and has no dependency on an
  LLM call — it's what query/pipeline.py wires in unconditionally for every
  query, so a query embedding an SSN or account number never reaches any
  LLM or the embedding endpoint, full stop.
- `detect_unstructured_pii` (names, addresses — CLAUDE.md's other stated
  requirement) needs a prompted-LLM call, which is a meaningfully different
  cost/latency/dependency profile. It's built and tested in isolation but
  NOT auto-wired into process_query's default path, so a caller opts into
  it explicitly (e.g. an intake step ahead of process_query) rather than
  every query silently paying for an extra LLM round-trip.

`sanitize_input` composes both when a caller wants the full guardrail.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ingestion.metadata.llm_client import LLMClient

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Coarse: SSNs written without dashes, and account/card-number-shaped runs.
# Deliberately broad (false positives redact something harmless; false
# negatives leak PII) -- 9-19 consecutive digits with no other structure.
_LONG_DIGIT_RUN_PATTERN = re.compile(r"\b\d{9,19}\b")

_UNSTRUCTURED_PII_SYSTEM_PROMPT = """Find any unstructured personally \
identifiable information in the text below: person names, home addresses, \
email addresses, phone numbers, or signatures. Do not flag company names, \
role titles, or generic legal terms. Respond with JSON only: {"pii_spans": \
[{"text": "...", "kind": "name"|"address"|"email"|"phone"|"signature"}]}"""


class PiiFinding(BaseModel):
    kind: str  # "ssn" | "account_number" | "name" | "address" | "email" | "phone" | "signature"
    matched_text: str
    detection_method: str  # "regex" | "llm"


class SanitizationResult(BaseModel):
    sanitized_text: str
    findings: list[PiiFinding]

    @property
    def any_pii_found(self) -> bool:
        return len(self.findings) > 0


def redact_structured_pii(text: str) -> SanitizationResult:
    findings: list[PiiFinding] = []

    def _replace_ssn(match: re.Match) -> str:
        findings.append(PiiFinding(kind="ssn", matched_text=match.group(0), detection_method="regex"))
        return "[REDACTED-SSN]"

    redacted = _SSN_PATTERN.sub(_replace_ssn, text)

    def _replace_long_digits(match: re.Match) -> str:
        findings.append(
            PiiFinding(kind="account_number", matched_text=match.group(0), detection_method="regex")
        )
        return "[REDACTED-NUMBER]"

    redacted = _LONG_DIGIT_RUN_PATTERN.sub(_replace_long_digits, redacted)

    return SanitizationResult(sanitized_text=redacted, findings=findings)


def detect_unstructured_pii(text: str, *, llm_client: LLMClient) -> list[PiiFinding]:
    result = llm_client.generate_json(_UNSTRUCTURED_PII_SYSTEM_PROMPT, text)
    findings: list[PiiFinding] = []
    for span in result.get("pii_spans", []):
        span_text = span.get("text")
        if not span_text:
            continue
        findings.append(
            PiiFinding(kind=span.get("kind", "unstructured"), matched_text=span_text, detection_method="llm")
        )
    return findings


def sanitize_input(text: str, *, llm_client: LLMClient | None = None) -> SanitizationResult:
    structured = redact_structured_pii(text)
    if llm_client is None:
        return structured

    sanitized_text = structured.sanitized_text
    unstructured_findings = detect_unstructured_pii(sanitized_text, llm_client=llm_client)
    for finding in unstructured_findings:
        if finding.matched_text in sanitized_text:
            sanitized_text = sanitized_text.replace(
                finding.matched_text, f"[REDACTED-{finding.kind.upper()}]"
            )

    return SanitizationResult(
        sanitized_text=sanitized_text, findings=structured.findings + unstructured_findings
    )
