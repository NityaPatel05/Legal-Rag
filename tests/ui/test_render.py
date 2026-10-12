"""Quote highlighting must mark exactly what the citation validator matched,
and never let contract or model text through as HTML."""

from render import clause_card_html, highlight_html, quote_span

CLAUSE = "10. Governing Law.\nThis Agreement is governed by the laws of the State of Delaware,\nwithout regard to conflicts principles."


def test_quote_found_across_line_breaks_case_and_trailing_period():
    quote = "this agreement is governed by the laws of the State of Delaware, without regard."
    start, end = quote_span(CLAUSE, quote)
    assert CLAUSE[start:end].startswith("This Agreement")
    assert CLAUSE[start:end].endswith("without regard")


def test_curly_quotes_and_dashes_match_straight_ones():
    text = "the “Services” — as defined — shall"
    assert quote_span(text, 'the "Services" - as defined - shall') == (0, len(text))


def test_missing_or_too_short_quote_is_not_highlighted():
    assert quote_span(CLAUSE, "the laws of California") is None
    assert quote_span(CLAUSE, "law") is None  # shorter than the validator's minimum
    assert highlight_html(CLAUSE, [("1", "the laws of California")]) == highlight_html(CLAUSE, [])


def test_highlight_escapes_everything_including_the_match():
    text = 'Payment <img src=x onerror=alert(1)> is due within thirty (30) days & "net".'
    out = highlight_html(text, [("2", "is due within thirty (30) days & \"net\"")])
    assert "<img" not in out and "&lt;img" in out
    assert '<mark class="q"><span class="qn">2</span>is due within thirty (30) days &amp;' in out


def test_overlapping_quotes_keep_the_first_and_clause_card_escapes_metadata():
    out = highlight_html(CLAUSE, [("1", "governed by the laws"), ("2", "the laws of the State")])
    assert out.count("<mark") == 1
    card = clause_card_html(
        {"clause_id": "d::clause-0001", "section_id": "10", "heading": "<b>Law</b>", "text": CLAUSE,
         "injection_suspected": True, "injection_markers": ["<script>"]},
        [("1", "governed by the laws")],
    )
    assert "<b>Law" not in card and "&lt;script&gt;" in card
    assert "cited" in card and "possible prompt injection" in card
