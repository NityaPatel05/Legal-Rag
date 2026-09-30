"""normalize_source_text (stored text) and normalize_preserving_offsets
(parser's detection view) -- see ingestion/parsing/text_normalize.py."""

import pytest

from ingestion.parsing.text_normalize import normalize_preserving_offsets, normalize_source_text

MESSY = (
    "﻿MASTER AGREEMENT\r\n\r\n1. Services and Fees\rVendor​ provides\f"
    "2. Term\vLicen­see １０ days ﬁle … Acme™ 　end"
)


def test_line_endings_become_lf():
    assert normalize_source_text("a\r\nb\rc\fd\ve f") == "a\nb\nc\nd\ne\nf"


def test_unicode_spaces_become_ascii_space():
    assert normalize_source_text("1. A B C D　E") == "1. A B C D E"


def test_zero_width_and_bom_are_removed():
    assert normalize_source_text("﻿Sec​tion‌ 1‍⁠ termi­nation") == "Section 1 termination"


def test_nfkc_folds_compatibility_forms():
    assert normalize_source_text("１０ ﬁle …") == "10 file ..."


def test_ascii_text_is_returned_unchanged():
    text = "1. Services\n\nVendor provides services."
    assert normalize_source_text(text) is text
    assert normalize_preserving_offsets(text) is text


def test_normalize_source_text_is_idempotent():
    once = normalize_source_text(MESSY)
    assert normalize_source_text(once) == once
    assert "\r" not in once and " " not in once and "​" not in once and "﻿" not in once


@pytest.mark.parametrize("text", [MESSY, "a\r\nb", "…™", "plain"])
def test_offset_preserving_view_keeps_length(text):
    assert len(normalize_preserving_offsets(text)) == len(text)


def test_offset_preserving_view_maps_crlf_to_space_newline():
    # Not "\n\n": a wrapped CRLF line must not look like a paragraph break.
    assert normalize_preserving_offsets("in\r\nSection 3") == "in \nSection 3"


def test_offset_preserving_view_is_identity_on_normalized_text():
    normalized = normalize_source_text(MESSY)
    assert normalize_preserving_offsets(normalized) == normalized
