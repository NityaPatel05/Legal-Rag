from pathlib import Path

from ingestion.parsing.clause_parser import parse_clauses

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


def _load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_parses_expected_clause_headings_in_order():
    clauses = parse_clauses(_load_fixture(), document_id="doc-1")

    headings = [(c.section_id, c.heading) for c in clauses]
    assert headings == [
        (None, None),  # preamble: title + intro paragraph
        ("1", "Definitions"),
        ("2", "Term"),
        ("2.1", "Initial Term"),
        ("2.2", "Renewal"),
        ("3", "Termination"),
        ("4", "Governing Law and Jurisdiction"),
    ]


def test_clause_boundaries_are_contiguous_non_overlapping_and_cover_full_text():
    raw_text = _load_fixture()
    clauses = parse_clauses(raw_text, document_id="doc-1")

    assert clauses[0].start_offset == 0
    assert clauses[-1].end_offset == len(raw_text)
    for prev, nxt in zip(clauses, clauses[1:]):
        assert prev.end_offset == nxt.start_offset

    for clause in clauses:
        assert clause.text == raw_text[clause.start_offset : clause.end_offset]


def test_clause_ids_are_unique_and_ordered():
    clauses = parse_clauses(_load_fixture(), document_id="doc-1")
    ids = [c.clause_id for c in clauses]
    assert ids == sorted(set(ids), key=ids.index)
    assert len(set(ids)) == len(ids)
    assert all(cid.startswith("doc-1::clause-") for cid in ids)


def test_naive_fixed_window_chunking_would_straddle_a_clause_boundary():
    """Demonstrates the exact failure mode CLAUDE.md #1 calls out: a
    fixed-size window can straddle two clauses. The structure-based parser
    doesn't have this problem — see the previous test."""
    raw_text = _load_fixture()
    clauses = parse_clauses(raw_text, document_id="doc-1")

    boundary = next(c.start_offset for c in clauses if c.heading == "Renewal")
    naive_window = raw_text[boundary - 250 : boundary + 120]

    assert "Initial Term" in naive_window or "2.1" in naive_window
    assert "Renewal" in naive_window
    # i.e. one naive window spans two distinct clauses — the parser's actual
    # output never does this (previous test asserts non-overlap).


def test_captioned_article_heading_pattern():
    text = "Preamble text.\n\nARTICLE IV - TERMINATION\n\nEither party may terminate for cause."
    clauses = parse_clauses(text, document_id="doc-2")

    assert clauses[0].heading is None
    assert clauses[1].section_id == "IV"
    assert clauses[1].heading == "- TERMINATION" or clauses[1].heading == "TERMINATION"


def test_captioned_section_colon_heading_pattern():
    text = "Section 3: Governing Law\n\nThis Agreement is governed by Delaware law."
    clauses = parse_clauses(text, document_id="doc-3")

    assert len(clauses) == 1
    assert clauses[0].section_id == "3"
    assert "Governing Law" in clauses[0].heading


def test_no_headings_returns_single_preamble_clause():
    text = "This is a short document with no numbered sections or captions at all."
    clauses = parse_clauses(text, document_id="doc-4")

    assert len(clauses) == 1
    assert clauses[0].heading is None
    assert clauses[0].section_id is None
    assert clauses[0].text == text


def test_allcaps_standalone_heading_creates_a_boundary():
    text = "Preamble text.\n\nDEFINITIONS\n\nThis section defines terms used elsewhere."
    clauses = parse_clauses(text, document_id="doc-5")

    assert len(clauses) == 2
    assert clauses[1].heading == "DEFINITIONS"
    assert clauses[1].section_id is None  # ALL-CAPS headings aren't numbered, no section_id


def test_allcaps_heading_at_document_start_is_treated_as_title_not_a_boundary():
    """The exact false-positive this pattern is riskiest for: a document
    whose very first line is its own all-caps title."""
    text = "MASTER SERVICES AGREEMENT\n\nThis Agreement is entered into by the parties."
    clauses = parse_clauses(text, document_id="doc-6")

    assert len(clauses) == 1
    assert clauses[0].heading is None


def test_allcaps_blocklist_excludes_exhibit_stubs():
    text = "Preamble text.\n\nEXHIBIT 10.6\n\nMore preamble content follows here."
    clauses = parse_clauses(text, document_id="doc-7")

    assert len(clauses) == 1  # no boundary introduced at the EXHIBIT stub


def test_lettered_subsection_creates_a_boundary_with_no_section_id():
    text = "1. Recitals\n\nA. Seller sells widgets.\n\nB. Buyer buys widgets."
    clauses = parse_clauses(text, document_id="doc-8")

    headings = [(c.section_id, c.heading) for c in clauses]
    assert headings == [
        ("1", "Recitals"),
        (None, "Seller sells widgets."),
        (None, "Buyer buys widgets."),
    ]


def test_lettered_subsections_reused_later_in_the_document_do_not_collide():
    """The real risk found by sampling CUAD data: 'A.'/'B.' gets reused for
    multiple unrelated enumerations in one document. section_id=None for all
    of them means there's nothing to collide on in a section_id -> clause
    lookup table (see clause_parser.py's module docstring)."""
    text = (
        "1. Recitals\n\nA. Seller sells widgets.\n\nB. Buyer buys widgets.\n\n"
        "2. Sale Guidelines\n\nA. The Sale shall be conducted in good faith.\n\n"
        "B. The Sale shall comply with local law."
    )
    clauses = parse_clauses(text, document_id="doc-9")

    lettered_clauses = [c for c in clauses if c.section_id is None and c.heading]
    assert len(lettered_clauses) == 4
    assert all(c.section_id is None for c in lettered_clauses)
    # all four remain distinct, non-overlapping clauses despite the letter reuse
    assert len({c.clause_id for c in lettered_clauses}) == 4


def test_indented_numbered_heading_creates_a_boundary():
    """R13: 64 of CUAD's 183 under-segmented documents indent their headings."""
    text = "Preamble.\n\n    1. Services\n\nVendor provides services.\n\n    2. Fees\n\nCustomer pays."
    clauses = parse_clauses(text, document_id="doc-10")

    assert [(c.section_id, c.heading) for c in clauses] == [
        (None, None),
        ("1", "Services"),
        ("2", "Fees"),
    ]


def test_flattened_text_headings_are_found_mid_line():
    """R13: text extracted with its newlines lost puts every heading mid-line."""
    text = (
        'This Agreement is made by Acme ("Licensor"). SECTION 1. GRANT Section 1.1 '
        "License. Licensor grants a license. Section 1.2 Sublicensing. Licensee may "
        "sublicense. 2. Term. The term is one year. 3. Termination. Either party may terminate."
    )
    clauses = parse_clauses(text, document_id="doc-11")

    assert [c.section_id for c in clauses] == [None, "1", "1.1", "1.2", "2", "3"]
    assert clauses[4].heading == "Term"
    assert clauses[1].text.startswith("SECTION 1.")
    # contiguous: no characters lost or duplicated at inline boundaries
    assert "".join(c.text for c in clauses) == text


def test_inline_cross_references_and_stray_numbers_are_not_headings():
    """Numbers that break the section sequence are rejected, so a mid-sentence
    'Section 12.3' reference or an amount does not start a clause."""
    text = (
        "Preamble. 1. Grant. Licensor grants a license, subject to Section 12.3 Licensee "
        "obligations. The fee is $5. 7 Parties may review it. 2. Term. One year."
    )
    clauses = parse_clauses(text, document_id="doc-12")

    assert [c.section_id for c in clauses] == [None, "1", "2"]


def test_inline_pass_is_skipped_when_line_headings_are_found():
    text = "1. Grant\n\nSee the terms. 7. Other text.\n\n2. Term\n\nOne year.\n\n3. Fees\n\nPaid."
    clauses = parse_clauses(text, document_id="doc-13")

    assert [c.section_id for c in clauses] == ["1", "2", "3"]
