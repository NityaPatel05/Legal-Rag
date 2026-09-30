"""Regression tests for clause-boundary bugs found in review: wrapped lines
read as headings (B7), invisible characters defeating the heading patterns
(B17), and passage packing with a tiny max_chars (L1)."""

import pytest

from ingestion.parsing.clause_parser import (
    MIN_PASSAGE_CHARS,
    parse_clauses,
    split_oversize_clauses,
)


def _ids(text: str) -> list[str | None]:
    return [c.section_id for c in parse_clauses(text, document_id="d")]


def _assert_contiguous(text: str) -> None:
    clauses = parse_clauses(text, document_id="d")
    assert "".join(c.text for c in clauses) == text
    for c in clauses:
        assert c.text == text[c.start_offset : c.end_offset]


# --- B7: wrapped continuation lines are not headings ---


def test_wrapped_section_cross_reference_is_not_a_heading():
    text = (
        "1. Payment\n\nFees are payable as set forth in\nSection 3 hereof. Late fees apply.\n\n"
        "2. Term\n\nOne year.\n\n3. Termination\n\nEither party may terminate."
    )
    clauses = parse_clauses(text, document_id="d")
    assert [c.section_id for c in clauses] == ["1", "2", "3"]
    assert clauses[2].heading == "Termination"  # the real Section 3 isn't shadowed
    _assert_contiguous(text)


def test_wrapped_number_at_line_start_is_not_a_heading():
    text = "1. Payment\n\nCustomer shall pay each invoice within\n30 Business Days of receipt.\n\n2. Term\n\nOne year."
    assert _ids(text) == ["1", "2"]


def test_out_of_sequence_number_after_a_blank_line_is_not_a_heading():
    text = (
        "1. Fees\n\nPaid monthly.\n\n2. Notice\n\nNotice must be written.\n\n"
        "30 Business Days notice is required for changes.\n\n3. Law\n\nDelaware."
    )
    assert _ids(text) == ["1", "2", "3"]


def test_headings_without_blank_lines_still_split_after_a_sentence_end():
    text = "1. Services\nVendor provides services.\n2. Fees\nCustomer pays.\n3. Term\nOne year."
    assert _ids(text) == ["1", "2", "3"]
    _assert_contiguous(text)


def test_heading_under_an_unmatched_title_line_still_splits():
    text = "1. Services\n\nVendor provides services.\n\nFees and Payment\n2.1 Customer pays monthly.\n2.2 Late fees apply."
    assert _ids(text) == ["1", "2.1", "2.2"]


def test_numbering_jump_is_kept_when_the_next_heading_confirms_it():
    text = "1. A\n\nx.\n\n2. B\n\nx.\n\n9. C\n\nx.\n\n10. D\n\nx."
    assert _ids(text) == ["1", "2", "9", "10"]


def test_exhibit_numbering_restart_is_kept():
    text = (
        "1. Scope\n\nx.\n\n2. Price\n\nx.\n\n3. Law\n\nx.\n\n4. Notices\n\nx.\n\n"
        "EXHIBIT A\n\n1. Deliverables\n\nx.\n\n2. Milestones\n\nx."
    )
    assert _ids(text) == ["1", "2", "3", "4", "1", "2"]


def test_article_numbering_does_not_reset_the_section_series():
    text = (
        "ARTICLE 1\n\nSection 1. Grant.\n\nSection 2. Scope.\n\nSection 3. Limits.\n\n"
        "ARTICLE 2\n\nSection 1. Fees.\n\nSection 2. Taxes."
    )
    assert _ids(text) == ["1", "1", "2", "3", "2", "1", "2"]


def test_wrapped_allcaps_disclaimer_is_not_one_heading_per_line():
    text = (
        "1. Warranty\n\nIN NO EVENT SHALL EITHER PARTY BE LIABLE\n"
        "FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES\nARISING OUT OF THIS AGREEMENT.\n\n2. Term\n\nOne year."
    )
    headings = [c.heading for c in parse_clauses(text, document_id="d")]
    assert "FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES" not in headings
    assert _ids(text)[-1] == "2"


def test_allcaps_heading_under_a_caption_still_splits():
    text = "Preamble.\n\nARTICLE 1\nDEFINITIONS\n\nTerms.\n\nARTICLE 2\nTERM\n\nOne year."
    headings = [c.heading for c in parse_clauses(text, document_id="d")]
    assert "DEFINITIONS" in headings and "TERM" in headings


# --- B17: invisible characters no longer defeat heading detection ---


def test_crlf_text_keeps_allcaps_headings_and_raw_offsets():
    text = "Preamble text.\r\n\r\nDEFINITIONS\r\n\r\nThis section defines terms used elsewhere."
    clauses = parse_clauses(text, document_id="d")
    assert [c.heading for c in clauses] == [None, "DEFINITIONS"]
    _assert_contiguous(text)


def test_crlf_does_not_make_wrapped_lines_look_like_paragraph_breaks():
    text = "1. Payment\r\n\r\nPay within\r\n30 Business Days of receipt.\r\n\r\n2. Term\r\n\r\nOne year."
    assert _ids(text) == ["1", "2"]


def test_nbsp_after_section_number_is_a_heading():
    text = "Preamble.\n\n1. Services\n\nVendor provides.\n\n2. Fees\n\nCustomer pays.\n\n3. Term\n\nOne year."
    assert _ids(text) == [None, "1", "2", "3"]
    _assert_contiguous(text)


def test_form_feed_before_heading_is_a_line_break():
    text = "Preamble.\n\n1. Services\n\nVendor provides.\f2. Fees\n\nCustomer pays.\n\n3. Term\n\nOne year."
    assert _ids(text) == [None, "1", "2", "3"]
    _assert_contiguous(text)


def test_zero_width_space_in_heading_is_ignored():
    text = "Preamble.\n\n1.​Services\n\nVendor provides.\n\n2. Fees\n\nPays.\n\n3. Term\n\nOne year."
    assert _ids(text) == [None, "1", "2", "3"]
    _assert_contiguous(text)


# --- L1: passage packing validates max_chars ---


@pytest.mark.parametrize("max_chars", [0, 1, 3, MIN_PASSAGE_CHARS - 1])
def test_tiny_max_chars_is_rejected_instead_of_looping(max_chars):
    clauses = parse_clauses("1. Big\n\n" + "word " * 500, document_id="d")
    with pytest.raises(ValueError, match="max_chars"):
        split_oversize_clauses(clauses, max_chars=max_chars)


def test_minimum_max_chars_splits_within_bounds():
    raw = "1. Big\n\n" + "word " * 500
    clauses = parse_clauses(raw, document_id="d")
    passages = split_oversize_clauses(clauses, max_chars=MIN_PASSAGE_CHARS)
    assert all(len(p.text) <= MIN_PASSAGE_CHARS for p in passages)
    assert "".join(p.text for p in passages) == raw
