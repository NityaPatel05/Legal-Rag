"""Coercion helpers for untrusted LLM output (CLAUDE.md invariant #11).

`bool(x)` is the wrong coercion for model JSON: bool("false") is True and
bool(None) is False, so a guardrail read as `bool(result.get("is_flagged"))`
fails OPEN on a missing field and flags on the string "false" (review
finding B9). Callers choose the default explicitly, so a security check can
fail closed while a metadata flag fails to False."""

from __future__ import annotations

_TRUE = frozenset({"true", "yes", "y", "1"})
_FALSE = frozenset({"false", "no", "n", "0"})


def as_bool(value, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return default


def as_label(value) -> str | None:
    """A closed-vocabulary label as the model spelled it, normalised for
    comparison ("Comparison " -> "comparison")."""
    if not isinstance(value, str):
        return None
    label = value.strip().lower().replace("-", "_").replace(" ", "_")
    return label or None


def as_text(value, *, max_chars: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_chars] or None
