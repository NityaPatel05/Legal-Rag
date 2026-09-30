"""Jurisdiction / governing-law filter — CLAUDE.md #2: two independent,
optional predicates, applied immediately after the permission filter. Each is
only added if the caller explicitly provided it; a jurisdiction-only query
must never silently also scope to governing_law, or vice versa.

Values are normalized with the same function ingestion uses to store them
(ingestion/metadata/normalize.py), so "State of Delaware" in a query matches
"delaware" in the index."""

from __future__ import annotations

from ingestion.metadata.normalize import normalize_place
from query.schemas import FilterCondition, QueryFilters


def build_jurisdiction_governing_law_filter(filters: QueryFilters) -> list[FilterCondition]:
    conditions: list[FilterCondition] = []
    jurisdiction = normalize_place(filters.jurisdiction)
    governing_law = normalize_place(filters.governing_law)
    if jurisdiction is not None:
        conditions.append(FilterCondition(field="jurisdiction", value=jurisdiction))
    if governing_law is not None:
        conditions.append(FilterCondition(field="governing_law", value=governing_law))
    return conditions


def build_document_filter(filters: QueryFilters) -> list[FilterCondition]:
    if filters.document_id is None:
        return []
    return [FilterCondition(field="document_id", value=filters.document_id)]
