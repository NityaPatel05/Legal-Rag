"""Ingestion-time scan for instruction-like text in contract clauses (roadmap
R10, review finding S5).

Contract text is counterparty-authored, and every clause is eventually
pasted into an LLM prompt (tagging, enrichment, generation). Until now the
only defence was the prompt-level trust label. This scan is deterministic
and cheap: it flags clauses whose text looks like it is addressing a model
rather than a contracting party. The flag is stored in the clause payload
(ClauseMetadata.injection_suspected / injection_markers), carried onto the
generation prompt as a stronger warning label, recorded as a retrieval
trace step and audited when a flagged clause is retrieved.

It does NOT remove or block the clause: a legitimate contract can contain
these words, and a flagged clause is still evidence the lawyer may need.
It is a signal for the reviewer and the prompt, not a filter.

Patterns are deliberately specific to model-directed phrasing. Contract
English uses "you are responsible for", "act as agent", "system" and
"instructions" constantly, so bare "you are" / "act as" / "instructions" are
NOT markers -- a scanner that flags every consumer agreement is ignored."""

from __future__ import annotations

import re
import unicodedata

_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for name, pattern in (
        (
            "override_instructions",
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your)\b"
            r"[^.\n]{0,20}\b(?:instructions?|prompts?|rules|directions)\b",
        ),
        ("new_instructions", r"\b(?:new|updated|revised)\s+(?:system\s+)?instructions\s*:"),
        ("model_address", r"\b(?:you are|act as|pretend to be|you're)\s+(?:now\s+)?(?:an?\s+)?"
         r"(?:ai|a\.i\.|language model|llm|chatbot|assistant|chatgpt|gpt|claude)\b"),
        ("model_address", r"\b(?:as an|dear)\s+(?:ai|language model|llm|assistant)\b"),
        ("prompt_reference", r"\b(?:system|developer)\s+(?:prompt|message)\b"),
        ("role_marker", r"^\s*(?:system|assistant|user|human)\s*:"),
        ("chat_template_token", r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>|\[/?INST\]|<</?SYS>>"),
        ("markdown_instruction_header", r"^\s*#{2,}\s*(?:instruction|system|prompt)s?\b"),
        (
            "output_manipulation",
            r"\b(?:respond|reply|answer|output)\s+(?:only\s+)?(?:with|that)\b[^.\n]{0,60}\b(?:json|valid|no conflict|"
            r"not (?:binding|enforceable))\b",
        ),
        ("concealment", r"\bdo not (?:tell|reveal|mention|disclose) (?:this|these|the user|anyone)\b[^.\n]{0,40}"
         r"\b(?:instruction|prompt|model|ai)\b"),
    )
)


def scan_for_injection(text: str) -> list[str]:
    """Names of the injection patterns found in `text`, sorted, deduplicated.
    Empty list = nothing instruction-like found."""
    text = _canonical(text)
    return sorted({name for name, pattern in _PATTERNS if pattern.search(text)})


_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def _canonical(text: str) -> str:
    """NFKC folds full-width / stylised letters ("ｉｇｎｏｒｅ") onto ASCII, and
    zero-width characters and soft hyphens are dropped, so "ig​nore
    previous instructions" can't slip past the patterns (review finding S4)."""
    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
