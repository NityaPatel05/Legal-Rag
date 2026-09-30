"""R7: controlled vocabularies and the place-name gazetteer."""

import json
from pathlib import Path

import pytest

from ingestion.metadata.normalize import normalize_place
from ingestion.metadata.vocabulary import (
    CLAUSE_TYPES,
    CONTRACT_TYPES,
    CUAD_CLAUSE_CATEGORIES,
    CUAD_CONTRACT_TYPES,
    normalize_clause_type,
    normalize_contract_type,
)

_CUAD = Path("eval/cuad_preflight/data/CUAD_v1.json")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("NY", "new york"),
        ("N.Y.", "new york"),
        ("New York, New York", "new york"),
        ("Southern District of New York", "new york"),
        ("Wilmington, Delaware", "delaware"),
        ("the courts located in Santa Clara County, California", "california"),
        ("Washington, D.C.", "district of columbia"),
        ("Washington", "washington"),
        ("West Virginia", "west virginia"),
        ("Ontario, Canada", "ontario"),
        ("Texas, U.S.A.", "texas"),
        ("laws of England and Wales", "england and wales"),
        ("People's Republic of China", "china"),
    ],
)
def test_gazetteer_resolves_synonyms_and_hierarchy(raw, expected):
    assert normalize_place(raw) == expected


def test_gazetteer_does_not_guess_between_two_states():
    # Two different states named: fall back to plain cleanup, never pick one.
    assert normalize_place("Delaware law; courts of New York") == "delaware law; courts of new york"


def test_abbreviations_are_not_matched_inside_running_text():
    # "in" / "or" are words, not Indiana / Oregon, unless they're the whole value.
    assert normalize_place("arbitration in geneva or zurich") == "arbitration in geneva or zurich"


def test_unknown_place_keeps_conservative_normalization():
    assert normalize_place("the Laws of Ruritania.") == "ruritania"


def test_vocabularies_have_cuad_sizes():
    assert len(CUAD_CLAUSE_CATEGORIES) == 41
    assert len(CUAD_CONTRACT_TYPES) == 25
    assert len(set(CLAUSE_TYPES)) == len(CLAUSE_TYPES)


@pytest.mark.skipif(not _CUAD.exists(), reason="CUAD_v1.json not present")
def test_clause_vocabulary_matches_cuad_category_ids():
    cuad = json.loads(_CUAD.read_text(encoding="utf-8"))
    ids = [qa["id"].split("__")[-1] for qa in cuad["data"][0]["paragraphs"][0]["qas"]]
    assert tuple(ids) == CUAD_CLAUSE_CATEGORIES


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Governing Law", "Governing Law"),
        ("governing law clause", "Governing Law"),
        ("Term and Termination", "Termination"),
        ("Indemnity", "Indemnification"),
        ("choice of law", "Governing Law"),
        ("something the model made up", "Other"),
        (None, None),
        ("  ", None),
    ],
)
def test_clause_type_mapping(raw, expected):
    result = normalize_clause_type(raw)
    assert result == expected
    assert result is None or result in CLAUSE_TYPES


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Master Services Agreement", "Service"),
        ("NDA", "Non-Disclosure"),
        ("Mutual Non-Disclosure Agreement", "Non-Disclosure"),
        ("WEB SITE HOSTING AGREEMENT", "Hosting"),
        ("PROMOTION AND DISTRIBUTION AGREEMENT", "Distributor"),
        ("Exclusive Reseller Agreement", "Reseller"),
        ("Joint Filing Agreement", "Other"),
    ],
)
def test_contract_type_mapping(raw, expected):
    result = normalize_contract_type(raw)
    assert result == expected
    assert result in CONTRACT_TYPES
