"""Jurisdiction / governing-law filter — CLAUDE.md #2: two independent,
optional predicates, applied immediately after the permission filter. Each is
only added if the caller explicitly provided it; a jurisdiction-only query
must never silently also scope to governing_law, or vice versa."""

from __future__ import annotations

from query.schemas import FilterCondition, QueryFilters


def build_jurisdiction_governing_law_filter(filters: QueryFilters) -> list[FilterCondition]:
    conditions: list[FilterCondition] = []
    if filters.jurisdiction is not None:
        conditions.append(FilterCondition(field="jurisdiction", value=filters.jurisdiction))
    if filters.governing_law is not None:
        conditions.append(FilterCondition(field="governing_law", value=filters.governing_law))
    return conditions
