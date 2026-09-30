from query.filters.jurisdiction_governing_law import build_jurisdiction_governing_law_filter
from query.schemas import QueryFilters


def test_both_fields_produce_independent_conditions():
    """The CLAUDE.md #2 non-negotiable: tested with a contract where
    jurisdiction and governing_law differ -- they must appear as two
    separate predicates, neither derived from the other."""
    filters = QueryFilters(jurisdiction="New York", governing_law="Delaware")
    conditions = build_jurisdiction_governing_law_filter(filters)

    fields = {c.field: c.value for c in conditions}
    assert fields == {"jurisdiction": "new york", "governing_law": "delaware"}


def test_jurisdiction_only_does_not_implicitly_scope_governing_law():
    filters = QueryFilters(jurisdiction="New York", governing_law=None)
    conditions = build_jurisdiction_governing_law_filter(filters)

    fields = [c.field for c in conditions]
    assert fields == ["jurisdiction"]


def test_governing_law_only_does_not_implicitly_scope_jurisdiction():
    filters = QueryFilters(jurisdiction=None, governing_law="Delaware")
    conditions = build_jurisdiction_governing_law_filter(filters)

    fields = [c.field for c in conditions]
    assert fields == ["governing_law"]


def test_neither_field_set_produces_no_conditions():
    filters = QueryFilters()
    assert build_jurisdiction_governing_law_filter(filters) == []
