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

What the regex layer catches, and why it's shaped the way it is. The
redaction runs on every prompt to the default provider (invariant #12) --
including whole contract clauses at ingestion -- and contract text is full
of numbers: section numbers ("12.3.4"), dates ("2024-01-15"), amounts
("$1,000,000.00"), counts ("thirty (30) days", "10,000 shares"). Redacting
those would break tagging and generation (the model can't quote an amount
it never saw), so every pattern is anchored on structure ordinary contract
numbers don't have, and the ambiguous shapes must pass a checksum:

- SSN: 3-2-4 digits. "-" as before, plus one consistent space or "." (the
  "123 45 6789" / "123.45.6789" forms the dash-only pattern missed); those
  two forms must also be a valid SSN (no 000/666/9xx area, 00 group, 0000
  serial).
- Card numbers written with separators ("4111 1111 1111 1111",
  "3782-822463-10005"): only if the Luhn check passes, so a space-grouped
  reference number isn't redacted.
- IBAN ("GB82 WEST 1234 5698 7654 32"): only if the ISO 13616 mod-97 check
  passes.
- Email addresses.
- Phone numbers: NANP with separators ("(555) 123-4567", "555.123.4567",
  "+1 555 123 4567") and "+"-prefixed international numbers with 8-15
  digits (E.164). Dates, section numbers and comma-grouped amounts don't
  have a 3-3-4 separated shape or a leading "+".
- Any 9-19 digit run with no separators (account/card numbers, SSNs without
  dashes) -- unchanged, coarse on purpose.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel

from ingestion.metadata.llm_client import LLMClient

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Space- or dot-separated SSNs: the same separator twice, valid ranges only.
_SSN_SPACED_PATTERN = re.compile(
    r"(?<![\d.-])(?!000|666|9\d\d)\d{3}([ .])(?!00)\d{2}\1(?!0000)\d{4}(?![\d-]|\.\d)"
)
# Card numbers with one consistent space/dash separator: 4-4-4-4 (plus an
# optional 1-3 digit group for 17-19 digit PANs), and 4-6-5 (Amex) / 4-6-4
# (Diners).
_CARD_SEPARATED_PATTERN = re.compile(
    r"(?<![\d-])(?:\d{4}([ -])\d{4}\1\d{4}\1\d{4}(?:\1\d{1,3})?|\d{4}([ -])\d{6}\2\d{4,5})(?![\d-])"
)
# IBAN: country code, 2 check digits, then 11-30 alphanumerics, optionally
# grouped by single spaces. Validated with mod-97 before redacting.
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# NANP: optional +1, area code (parenthesised, or followed by a separator),
# then 3-sep-4. Unseparated 10-digit runs fall to the digit-run rule.
_PHONE_NANP_PATTERN = re.compile(
    r"(?<![\w.+-])(?:\+?1[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w-]|\.\d)"
)
# International: "+", country code, then 2-5 separated groups; the 8-15
# digit total is checked in code.
_PHONE_INTL_PATTERN = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?![\w-]|\.\d)")
# Coarse: SSNs written without dashes, and account/card-number-shaped runs.
# Deliberately broad (false positives redact something harmless; false
# negatives leak PII) -- 9-19 consecutive digits with no other structure.
_LONG_DIGIT_RUN_PATTERN = re.compile(r"\b\d{9,19}\b")

_UNSTRUCTURED_PII_SYSTEM_PROMPT = """Find any unstructured personally \
identifiable information in the text below: person names, home addresses, \
email addresses, phone numbers, or signatures. Do not flag company names, \
role titles, or generic legal terms. Respond with JSON only: {"pii_spans": \
[{"text": "...", "kind": "name"|"address"|"email"|"phone"|"signature"}]}"""


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) checksum, used by every major card network."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    """ISO 13616 / ISO 7064 mod-97-10: move the first four characters to
    the end, map letters to 10..35, and the number mod 97 must be 1."""
    iban = candidate.replace(" ", "")
    if not 15 <= len(iban) <= 34:
        return False
    rotated = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in rotated)) % 97 == 1


def _card_valid(candidate: str) -> bool:
    return luhn_valid(re.sub(r"\D", "", candidate))


def _phone_intl_valid(candidate: str) -> bool:
    return 8 <= sum(ch.isdigit() for ch in candidate) <= 15


# (kind, pattern, replacement, validator). Order matters: the specific,
# checksummed shapes run before the coarse digit-run fallback.
_STRUCTURED_RULES: list[tuple[str, re.Pattern[str], str, Callable[[str], bool] | None]] = [
    ("ssn", _SSN_PATTERN, "[REDACTED-SSN]", None),
    ("ssn", _SSN_SPACED_PATTERN, "[REDACTED-SSN]", None),
    ("card_number", _CARD_SEPARATED_PATTERN, "[REDACTED-CARD]", _card_valid),
    ("iban", _IBAN_PATTERN, "[REDACTED-IBAN]", iban_valid),
    ("email", _EMAIL_PATTERN, "[REDACTED-EMAIL]", None),
    ("phone", _PHONE_NANP_PATTERN, "[REDACTED-PHONE]", None),
    ("phone", _PHONE_INTL_PATTERN, "[REDACTED-PHONE]", _phone_intl_valid),
    ("account_number", _LONG_DIGIT_RUN_PATTERN, "[REDACTED-NUMBER]", None),
]


class PiiFinding(BaseModel):
    # regex: "ssn" | "card_number" | "iban" | "email" | "phone" | "account_number"
    # llm:   "name" | "address" | "email" | "phone" | "signature"
    kind: str
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
    redacted = text
    for kind, pattern, replacement, validator in _STRUCTURED_RULES:

        def _replace(match: re.Match[str], kind=kind, replacement=replacement, validator=validator) -> str:
            matched = match.group(0)
            if validator is not None and not validator(matched):
                return matched
            findings.append(PiiFinding(kind=kind, matched_text=matched, detection_method="regex"))
            return replacement

        redacted = pattern.sub(_replace, redacted)

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
